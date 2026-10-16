#!/usr/bin/env python3
"""RF-DETR object-detection inference CLI."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_MAP = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "large": RFDETRLarge,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR object detection on an image folder or a video.",
    )
    parser.add_argument("--folder", type=Path, help="Folder of images.")
    parser.add_argument("--video", action="store_true", help="Process a video file.")
    parser.add_argument("--video-path", type=Path, help="Video file, used with --video.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint_best_total.pth")
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        help="class_names.txt (one name per line). Defaults to the file next to the checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "output" / "runs",
        help="Base directory for annotated outputs.",
    )
    parser.add_argument("--model-size", choices=sorted(MODEL_MAP), default="medium")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Compile the model for faster repeated inference.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str | None:
    if args.video:
        if args.folder:
            return "use either --folder or --video, not both"
        if not args.video_path:
            return "--video-path is required when using --video"
        return None
    if args.video_path:
        return "--video-path can only be used together with --video"
    if not args.folder:
        return "specify --folder or --video with --video-path"
    return None


def load_class_names(path: Path) -> dict[int, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [line for line in lines if line]
    if not names:
        raise ValueError(f"no class names in {path}")
    return {index: name for index, name in enumerate(names)}


def collect_images(folder: Path) -> list[Path]:
    images = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")
    return images


def build_labels(detections: sv.Detections, class_names: dict[int, str]) -> list[str]:
    return [
        f"{class_names.get(int(class_id), str(class_id))} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]


def annotate(frame: np.ndarray, detections: sv.Detections, labels: list[str]) -> np.ndarray:
    annotated = sv.BoxAnnotator().annotate(frame.copy(), detections)
    if labels:
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
    return annotated


def load_model(args: argparse.Namespace) -> Any:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = MODEL_MAP[args.model_size](pretrain_weights=str(checkpoint))
    if args.optimize:
        model.optimize_for_inference(compile=True, batch_size=1)
    return model


def run_on_folder(args, model, class_names) -> Path:
    folder = args.folder.expanduser().resolve()
    images = collect_images(folder)
    now = datetime.now()
    run_dir = args.output.expanduser().resolve() / f"{now:%y-%m-%d}_{now:%H%M}_{folder.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Images : {len(images)}")
    print(f"Output : {run_dir}")
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        frame = np.array(image)
        detections = model.predict(image, threshold=args.threshold)
        labels = build_labels(detections, class_names)
        out = run_dir / f"{image_path.stem}_annotated.jpg"
        Image.fromarray(annotate(frame, detections, labels)).save(out)
        print(f"  {image_path.name} -> {len(detections)} -> {out.name}")
    return run_dir


def run_on_video(args, model, class_names) -> Path:
    video_path = args.video_path.expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    now = datetime.now()
    run_dir = args.output.expanduser().resolve() / f"{now:%y-%m-%d}_{now:%H%M}_{video_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")
    print(f"Output : {output_path}")
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=args.threshold)
            labels = build_labels(detections, class_names)
            writer.write(annotate(frame_bgr, detections, labels))
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  processed {frame_idx} frames", flush=True)
    finally:
        capture.release()
        writer.release()
    print(f"  done -> {frame_idx} frames")
    return run_dir


def main() -> int:
    args = parse_args()
    error = validate_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    checkpoint = args.checkpoint.expanduser().resolve()
    classes_path = (
        args.classes.expanduser().resolve()
        if args.classes
        else checkpoint.parent / "class_names.txt"
    )
    if not classes_path.is_file():
        print(f"Error: class list not found: {classes_path}", file=sys.stderr)
        return 1
    try:
        class_names = load_class_names(classes_path)
        model = load_model(args)
        print(f"Checkpoint : {checkpoint}")
        print(f"Classes    : {', '.join(class_names[i] for i in range(len(class_names)))}")
        if args.video:
            run_on_video(args, model, class_names)
        else:
            run_on_folder(args, model, class_names)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
