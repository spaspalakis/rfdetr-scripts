#!/usr/bin/env python3
"""Train an RF-DETR detector on custom classes from a Roboflow COCO export.

Tested with rfdetr==1.11.0. The dataset goes to RF-DETR as it is:

    <dataset>/train/_annotations.coco.json  (+ images)
    <dataset>/valid/_annotations.coco.json  (+ images)
    <dataset>/test/_annotations.coco.json   (+ images, optional)

RF-DETR 1.11 drops the Roboflow project category (the parent category with
no boxes), so the model classes are ids 0..N-1 in the COCO category order.
No COCO-90 classes are used. The checkpoint works with class_names.txt and a
class id offset of 0.

data.yaml:

    path: /data/reaction-land_model.v1i.coco
    names:          # optional check: must match the dataset classes, same order
      0: boat
      1: person
      2: vehicle
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
COCO_JSON = "_annotations.coco.json"
DEFAULT_RESOLUTION = {"nano": 384, "small": 512, "medium": 576, "large": 704}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR on a Roboflow COCO dataset with custom classes.")
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


def read_names(names, source: Path) -> list[str]:
    """Accept ``[a, b]`` or ``{0: a, 1: b}`` and return names in id order."""
    if isinstance(names, list):
        return [str(name).strip() for name in names]
    if isinstance(names, dict):
        try:
            by_id = {int(key): str(value).strip() for key, value in names.items()}
        except ValueError:
            raise ValueError(f"{source}: 'names' keys must be integers 0..N-1")
        if sorted(by_id) != list(range(len(by_id))):
            raise ValueError(f"{source}: 'names' keys must be 0..N-1 with no gaps, got {sorted(by_id)}")
        return [by_id[index] for index in range(len(by_id))]
    raise ValueError(f"{source}: 'names' must be a list or a map of 0..N-1")


def load_data_yaml(path: Path) -> tuple[Path, list[str] | None]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"data yaml not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not data.get("path"):
        raise ValueError(f"{path} must define 'path', the dataset folder")
    root = Path(str(data["path"])).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve()
    names = read_names(data["names"], path) if data.get("names") else None
    return root, names


def check_dataset(root: Path) -> list[str]:
    """Check the Roboflow COCO layout and print boxes per class for each split."""
    from rfdetr.detr import RFDETR

    if not root.is_dir():
        raise ValueError(f"dataset folder not found: {root}")
    for split in ("train", "valid"):
        if not (root / split / COCO_JSON).is_file():
            raise ValueError(f"{root / split / COCO_JSON} not found. Download the dataset in COCO format.")

    # The same class list RF-DETR builds for the model (project category removed).
    classes = list(RFDETR._load_classes(str(root)))
    print("Dataset classes (model ids):")
    for index, name in enumerate(classes):
        print(f"  {index}  {name}")

    per_split = {}
    for split in ("train", "valid", "test"):
        ann_file = root / split / COCO_JSON
        if not ann_file.is_file():
            continue
        data = json.loads(ann_file.read_text(encoding="utf-8"))
        names = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
        counts = Counter(names.get(ann["category_id"], "?") for ann in data.get("annotations", []))
        per_split[split] = counts
        print(f"  {split:<5}: {len(data.get('images', []))} images, {sum(counts.values())} boxes")

    print("Boxes per class (" + " / ".join(per_split) + "):")
    for index, name in enumerate(classes):
        print(f"  {index}  {name:<20} " + " / ".join(f"{per_split[s][name]:>6}" for s in per_split))
    empty = [name for name in classes if per_split["train"][name] == 0]
    if empty:
        raise ValueError(f"no training boxes for {empty}")
    return classes


def main() -> int:
    args = parse_args()
    rfdetr_version = metadata.version("rfdetr")
    if rfdetr_version != TESTED_RFDETR:
        print(f"Warning: rfdetr {rfdetr_version} is installed, this script was tested on {TESTED_RFDETR}.")

    try:
        root, expected = load_data_yaml(args.data)
        print(f"Dataset   : {root}")
        classes = check_dataset(root)
        if expected is not None and expected != classes:
            raise ValueError(
                f"'names' in {args.data} is {expected}, the dataset classes are {classes}. "
                "Fix data.yaml (or the dataset) so both list the same classes in the same order."
            )
    except (OSError, ValueError, KeyError, yaml.YAMLError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

    model_cls = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}[args.model_size]
    resolution = args.resolution or DEFAULT_RESOLUTION[args.model_size]
    run_id = args.run_id or f"{datetime.now():%y-%m-%d_%H-%M}_det_{args.model_size}_r{resolution}"
    run_dir = (args.output / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "class_names.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    print(f"Model     : RFDETR{args.model_size.capitalize()} r{resolution} (rfdetr {rfdetr_version})")
    print(f"Batch     : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"Output    : {run_dir}")

    # Starts from RF-DETR's pretrained weights; the class head is rebuilt for the dataset classes.
    model = model_cls()
    model.train(
        dataset_dir=str(root),
        dataset_file="roboflow",
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
        run_test=(root / "test" / COCO_JSON).is_file(),
        log_per_class_metrics=True,
    )

    # RF-DETR prints per-class AP 50:95 only; score the best checkpoint for AP50 per class too.
    best = run_dir / "checkpoint_best_total.pth"
    class_ap_report = None
    if best.is_file():
        from class_ap import format_table, per_class_ap

        split = "test" if (root / "test" / COCO_JSON).is_file() else "val"
        print(f"\nScoring AP50 per class on {split}...", flush=True)
        best_model = model_cls(pretrain_weights=str(best), resolution=resolution)
        class_ap_report = per_class_ap(best_model, root, split, classes)
        table = "\n".join(format_table(class_ap_report))
        print(table, flush=True)
        (run_dir / "class_ap.txt").write_text(table + "\n", encoding="utf-8")

    (run_dir / "train_info.json").write_text(
        json.dumps(
            {
                "rfdetr_version": rfdetr_version,
                "model_size": args.model_size,
                "resolution": resolution,
                "class_names": classes,
                "class_id_offset": 0,
                "dataset": str(root),
                "class_ap": class_ap_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Checkpoint: {run_dir / 'checkpoint_best_total.pth'}")
    print(f"Copy checkpoint_best_total.pth and class_names.txt to models/<land|sea>/det_{args.model_size}_r{resolution}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
