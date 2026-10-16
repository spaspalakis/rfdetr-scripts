#!/usr/bin/env python3
"""RF-DETR segmentation inference CLI."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegSmall

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_BASE_OUTPUT = Path("/home/spaspalakis/Documents/rf-detr/output")
DEFAULT_CHECKPOINT = DEFAULT_BASE_OUTPUT / "models/26-06-02_2344_seg_small_b2_ga8_e200/checkpoint_best_total.pth"
DEFAULT_ANNOTATIONS = Path(
    "/home/spaspalakis/Documents/iDriving/uc2.2/"
    "UC2.2-Instance_segmentation.v1i.coco/train/_annotations.coco.json"
)

MODEL_MAP = {
    "small": RFDETRSegSmall,
    "medium": RFDETRSegMedium,
    "large": RFDETRSegLarge,
}

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR segmentation inference on images or video.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Path to the input folder containing images.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Process a video file instead of an image folder.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        help="Path to the input video file (required with --video).",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Draw bounding boxes on the output.",
    )
    parser.add_argument(
        "--polygon",
        action="store_true",
        help="Draw segmentation masks (polygons) on the output.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to the trained checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help="COCO annotations JSON used to resolve class names.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BASE_OUTPUT / "runs",
        help=f"Base directory for inference runs (default: {DEFAULT_BASE_OUTPUT / 'runs'}).",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_MAP),
        default="small",
        help="RF-DETR segmentation model size (default: small).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Call model.optimize_for_inference() for faster repeated inference.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str | None:
    if not args.bbox and not args.polygon:
        return "specify at least one of --bbox or --polygon"

    if args.video:
        if args.folder:
            return "use either --folder or --video, not both"
        if not args.video_path:
            return "--video-path is required when using --video"
        return None

    if args.video_path:
        return "--video-path can only be used together with --video"

    if not args.folder:
        return "specify --folder for image inference or --video with --video-path for video inference"

    return None


def resolve_input_path(path: Path, *, must_be: str) -> Path:
    """Resolve user paths against cwd and the script directory."""
    expanded = path.expanduser()
    candidates: list[Path] = []

    if expanded.is_absolute():
        candidates.append(expanded)
        if len(expanded.parts) > 1:
            # Handles mistaken paths like /video_in/7.mp4 -> ./video_in/7.mp4
            relative_tail = Path(*expanded.parts[1:])
            candidates.extend((Path.cwd() / relative_tail, SCRIPT_DIR / relative_tail))
    else:
        candidates.extend((Path.cwd() / expanded, SCRIPT_DIR / expanded))

    checker = {"file": Path.is_file, "dir": Path.is_dir}[must_be]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if checker(resolved):
            return resolved

    checked = "\n  ".join(str(candidate.resolve()) for candidate in candidates)
    hint = (
        f"Use a relative path such as video_in/7.mp4 when running from {SCRIPT_DIR}, "
        f"or the full path {SCRIPT_DIR / 'video_in' / path.name}."
    )
    raise FileNotFoundError(
        f"{must_be.capitalize()} not found for input path: {path}\n"
        f"Checked:\n  {checked}\n"
        f"Hint: {hint}"
    )


def load_class_names(annotations_path: Path) -> dict[int, str]:
    with annotations_path.open() as f:
        coco = json.load(f)
    return {category["id"]: category["name"] for category in coco["categories"]}


def collect_images(folder: Path) -> list[Path]:
    images = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {folder}")
    return images


def build_run_dir(output_base: Path, source_name: str) -> Path:
    now = datetime.now()
    run_dir = output_base / f"{now:%y-%m-%d}_{now:%H%M}_{source_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_labels(detections: sv.Detections, class_names: dict[int, str]) -> list[str]:
    return [
        f"{class_names.get(class_id, str(class_id))} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]


def make_annotators(
    draw_bbox: bool,
    draw_polygon: bool,
) -> tuple[sv.MaskAnnotator | None, sv.BoxAnnotator | None, sv.LabelAnnotator]:
    mask_annotator = sv.MaskAnnotator() if draw_polygon else None
    box_annotator = sv.BoxAnnotator() if draw_bbox else None
    label_annotator = sv.LabelAnnotator()
    return mask_annotator, box_annotator, label_annotator


def annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    labels: list[str],
    mask_annotator: sv.MaskAnnotator | None,
    box_annotator: sv.BoxAnnotator | None,
    label_annotator: sv.LabelAnnotator,
) -> np.ndarray:
    annotated = frame.copy()

    if mask_annotator is not None:
        annotated = mask_annotator.annotate(annotated, detections)
    if box_annotator is not None:
        annotated = box_annotator.annotate(annotated, detections)
    if labels:
        annotated = label_annotator.annotate(annotated, detections, labels)

    return annotated


def load_model(args: argparse.Namespace) -> Any:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model_cls = MODEL_MAP[args.model_size]
    model = model_cls(pretrain_weights=str(checkpoint))
    if args.optimize:
        model.optimize_for_inference(compile=True, batch_size=1)
    return model


def run_on_folder(
    args: argparse.Namespace,
    model: Any,
    class_names: dict[int, str],
    mask_annotator: sv.MaskAnnotator | None,
    box_annotator: sv.BoxAnnotator | None,
    label_annotator: sv.LabelAnnotator,
) -> Path:
    input_folder = resolve_input_path(args.folder, must_be="dir")

    images = collect_images(input_folder)
    run_dir = build_run_dir(args.output.expanduser().resolve(), input_folder.name)

    print(f"Input folder : {input_folder}")
    print(f"Images found : {len(images)}")
    print(f"Visualization: bbox={args.bbox}, polygon={args.polygon}")
    print(f"Output dir   : {run_dir}")

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        frame_rgb = np.array(image)
        detections = model.predict(image, threshold=args.threshold)
        labels = build_labels(detections, class_names)

        annotated_rgb = annotate_frame(
            frame_rgb,
            detections,
            labels,
            mask_annotator,
            box_annotator,
            label_annotator,
        )

        output_path = run_dir / f"{image_path.stem}_annotated.jpg"
        Image.fromarray(annotated_rgb).save(output_path)
        print(f"  {image_path.name} -> {len(detections)} detections -> {output_path.name}")

    return run_dir


def run_on_video(
    args: argparse.Namespace,
    model: Any,
    class_names: dict[int, str],
    mask_annotator: sv.MaskAnnotator | None,
    box_annotator: sv.BoxAnnotator | None,
    label_annotator: sv.LabelAnnotator,
) -> Path:
    video_path = resolve_input_path(args.video_path, must_be="file")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    run_dir = build_run_dir(args.output.expanduser().resolve(), video_path.stem)
    output_path = run_dir / f"{video_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    print(f"Input video  : {video_path}")
    print(f"Frames       : {total_frames if total_frames > 0 else 'unknown'} @ {fps:.2f} fps")
    print(f"Visualization: bbox={args.bbox}, polygon={args.polygon}")
    print(f"Output video : {output_path}")

    frame_idx = 0
    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = model.predict(frame_rgb, threshold=args.threshold)
            labels = build_labels(detections, class_names)

            annotated_bgr = annotate_frame(
                frame_bgr,
                detections,
                labels,
                mask_annotator,
                box_annotator,
                label_annotator,
            )
            writer.write(annotated_bgr)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  processed {frame_idx} frames", flush=True)
    finally:
        capture.release()
        writer.release()

    print(f"  done -> {frame_idx} frames -> {output_path.name}")
    return run_dir


def main() -> int:
    args = parse_args()

    error = validate_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    annotations = args.annotations.expanduser().resolve()
    if not annotations.is_file():
        print(f"Error: annotations file not found: {annotations}", file=sys.stderr)
        return 1

    try:
        class_names = load_class_names(annotations)
        model = load_model(args)
        mask_annotator, box_annotator, label_annotator = make_annotators(args.bbox, args.polygon)

        print(f"Checkpoint   : {args.checkpoint.expanduser().resolve()}")

        if args.video:
            run_on_video(
                args,
                model,
                class_names,
                mask_annotator,
                box_annotator,
                label_annotator,
            )
        else:
            run_on_folder(
                args,
                model,
                class_names,
                mask_annotator,
                box_annotator,
                label_annotator,
            )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
