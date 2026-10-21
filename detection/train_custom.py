#!/usr/bin/env python3
"""Train an RF-DETR detector on custom classes from a YOLO data.yaml.

Tested with rfdetr==1.11.0. The class list comes only from ``names`` in the
data.yaml. Class ids are 0..N-1 in that order, with no COCO classes and no
Roboflow project category. The checkpoint can be read by any code that uses
class_names.txt with a class id offset of 0.

data.yaml:

    path: /data/my_dataset      # dataset root (relative paths resolve from this yaml)
    train: train/images         # labels are found by swapping images -> labels
    val: valid/images           # "valid" is accepted too
    test: test/images           # optional
    names:
      0: boat
      1: car
      2: human

Each label file is YOLO format: ``class_id cx cy w h`` (normalized), one
object per line, same stem as the image.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TESTED_RFDETR = "1.11.0"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_RESOLUTION = {"nano": 384, "small": 512, "medium": 576, "large": 704}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR on custom classes from a YOLO data.yaml.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data.yaml", help="Dataset yaml (default: data.yaml next to this script).")
    parser.add_argument("--model-size", choices=sorted(DEFAULT_RESOLUTION), default="medium")
    parser.add_argument("--resolution", type=int, default=None, help="Input size, divisible by 32. Default depends on --model-size.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Effective batch = batch-size x grad-accum.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs.")
    parser.add_argument("--skip-epochs", type=int, default=0, help="Ignore the first N epochs when picking the best checkpoint.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "output", help="Folder for training runs.")
    parser.add_argument("--run-id", default=None, help="Run folder name. Default: <date>_det_<size>_r<resolution>.")
    return parser.parse_args()


# --- data.yaml ---------------------------------------------------------------

def read_names(names, source: Path) -> list[str]:
    """Accept ``[a, b]`` or ``{0: a, 1: b}`` and return names in id order."""
    if isinstance(names, list):
        ordered = [str(name).strip() for name in names]
    elif isinstance(names, dict):
        try:
            by_id = {int(key): str(value).strip() for key, value in names.items()}
        except ValueError:
            raise ValueError(f"{source}: 'names' keys must be integers 0..N-1")
        if sorted(by_id) != list(range(len(by_id))):
            raise ValueError(f"{source}: 'names' keys must be 0..N-1 with no gaps, got {sorted(by_id)}")
        ordered = [by_id[index] for index in range(len(by_id))]
    else:
        raise ValueError(f"{source}: 'names' must be a list or a map of 0..N-1")
    if not ordered or any(not name for name in ordered):
        raise ValueError(f"{source}: 'names' has an empty class name")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{source}: 'names' has duplicates: {ordered}")
    return ordered


def labels_dir_for(images_dir: Path) -> Path:
    """YOLO convention: .../images/... -> .../labels/..., else <dir>/labels."""
    parts = list(images_dir.parts)
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts)
    return images_dir / "labels"


def resolve_split(root: Path, value, split: str) -> tuple[Path, Path] | None:
    if not value:
        return None
    images_dir = Path(str(value)).expanduser()
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    images_dir = images_dir.resolve()
    if images_dir.is_dir() and (images_dir / "images").is_dir():
        images_dir = images_dir / "images"
    if not images_dir.is_dir():
        raise ValueError(f"{split}: images folder not found: {images_dir}")
    labels_dir = labels_dir_for(images_dir)
    if not labels_dir.is_dir():
        raise ValueError(f"{split}: labels folder not found: {labels_dir}")
    return images_dir, labels_dir


def load_data_yaml(path: Path) -> tuple[list[str], dict[str, tuple[Path, Path]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"data yaml not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping with path, train, val and names")
    if "names" not in data:
        raise ValueError(f"{path} must define 'names'")
    names = read_names(data["names"], path)

    root = Path(str(data.get("path") or path.parent)).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root not found: {root} (from 'path' in {path})")

    splits = {
        "train": resolve_split(root, data.get("train", "train/images"), "train"),
        "valid": resolve_split(root, data.get("val", data.get("valid", "valid/images")), "val"),
        "test": resolve_split(root, data.get("test"), "test"),
    }
    if splits["train"] is None or splits["valid"] is None:
        raise ValueError(f"{path} needs both a train and a val split")
    return names, {key: value for key, value in splits.items() if value is not None}


def check_split(split: str, images_dir: Path, labels_dir: Path, num_classes: int) -> Counter:
    """Count objects per class and stop on a class id outside 0..N-1."""
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise ValueError(f"{split}: no images in {images_dir}")
    counts: Counter = Counter()
    unlabeled = 0
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            unlabeled += 1
            continue
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(float(fields[0]))
            except ValueError:
                raise ValueError(f"{label}:{line_no}: bad class id {fields[0]!r}")
            if not 0 <= class_id < num_classes:
                raise ValueError(
                    f"{label}:{line_no}: class id {class_id} is outside 0..{num_classes - 1}. "
                    "The labels and 'names' in data.yaml do not match."
                )
            counts[class_id] += 1
    print(f"  {split:<5}: {len(images)} images, {unlabeled} without a label file (background)")
    return counts


def stage_dataset(run_dir: Path, names: list[str], splits: dict[str, tuple[Path, Path]]) -> Path:
    """Link the splits into the layout RF-DETR reads: <split>/images, <split>/labels, data.yaml."""
    staged = run_dir / "dataset"
    for split, (images_dir, labels_dir) in splits.items():
        target = staged / split
        target.mkdir(parents=True, exist_ok=True)
        for name, source in (("images", images_dir), ("labels", labels_dir)):
            link = target / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(source, target_is_directory=True)
    (staged / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(names), "names": names}, sort_keys=False), encoding="utf-8"
    )
    return staged


# --- main --------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    try:
        names, splits = load_data_yaml(args.data)
        print(f"Data yaml : {args.data.resolve()}")
        print("Classes   :")
        for index, name in enumerate(names):
            print(f"  {index}  {name}")
        print("Splits    :")
        totals = {}
        for split, (images_dir, labels_dir) in splits.items():
            totals[split] = check_split(split, images_dir, labels_dir, len(names))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Objects per class (train / valid):")
    for index, name in enumerate(names):
        print(f"  {index}  {name:<20} {totals['train'][index]:>7} / {totals['valid'][index]:>7}")
    missing = [names[index] for index in range(len(names)) if totals["train"][index] == 0]
    if missing:
        print(f"Error: no training objects for {missing}. Fix the labels or remove those names.", file=sys.stderr)
        return 1

    rfdetr_version = metadata.version("rfdetr")
    if rfdetr_version != TESTED_RFDETR:
        print(f"Warning: rfdetr {rfdetr_version} is installed, this script was tested on {TESTED_RFDETR}.")

    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    model_cls = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}[args.model_size]
    resolution = args.resolution or DEFAULT_RESOLUTION[args.model_size]
    run_id = args.run_id or f"{datetime.now():%y-%m-%d_%H-%M}_det_{args.model_size}_r{resolution}"
    run_dir = (args.output / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dataset(run_dir, names, splits)
    (run_dir / "class_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    print(f"Model     : RFDETR{args.model_size.capitalize()} r{resolution} (rfdetr {rfdetr_version})")
    print(f"Batch     : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"Output    : {run_dir}")

    # Starts from RF-DETR's pretrained weights; the class head is rebuilt for len(names) classes.
    model = model_cls(num_classes=len(names))
    model.train(
        dataset_dir=str(staged),
        dataset_file="yolo",
        class_names=names,
        output_dir=str(run_dir),
        resolution=resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        device=args.device,
        num_workers=args.workers,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        skip_best_epochs=args.skip_epochs,
        resume=str(args.resume) if args.resume else None,
        seed=args.seed,
        run_test="test" in splits,
        log_per_class_metrics=True,
    )

    best = run_dir / "checkpoint_best_total.pth"
    (run_dir / "train_info.json").write_text(
        json.dumps(
            {
                "rfdetr_version": rfdetr_version,
                "model_size": args.model_size,
                "resolution": resolution,
                "class_names": names,
                "class_id_offset": 0,
                "data_yaml": str(args.data.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Checkpoint: {best}")
    print(f"Copy checkpoint_best_total.pth and class_names.txt to models/<land|sea>/det_{args.model_size}_r{resolution}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
