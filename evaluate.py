#!/usr/bin/env python3
"""Re-evaluate a saved RF-DETR segmentation checkpoint.

Prints overall mAP and a per-class table, then writes the same report next
to the checkpoint so the numbers are not lost.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = Path(
    "/home/spaspalakis/Documents/iDriving/datasets/UC2.2-v6i.coco-segmentation"
)
DEFAULT_CHECKPOINT = Path(
    "/home/spaspalakis/Documents/rfdetr/output/"
    "26-09-27_23:26_seg_medium_r432_b2_ga8_e200/checkpoint_best_total.pth"
)

MODEL_MAP = {
    "nano": RFDETRSegNano,
    "small": RFDETRSegSmall,
    "medium": RFDETRSegMedium,
    "large": RFDETRSegLarge,
}

DEFAULT_RESOLUTION = {
    "nano": 312,
    "small": 384,
    "medium": 432,
    "large": 504,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure overall and per-class mAP for a saved RF-DETR checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint to evaluate (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset root in COCO or YOLO layout (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_MAP),
        default="medium",
        help="Model size used for this checkpoint (default: medium).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input resolution. Defaults: nano=312, small=384, medium=432, large=504.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to score (default: val).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Evaluation batch size (default: 2).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Data loader workers (default: 2).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
        help="Device used for evaluation (default: cuda).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report file. Defaults to eval_report.txt next to the checkpoint.",
    )
    return parser.parse_args()


def _fmt(value: float | None) -> str:
    if value is None or value != value or value < 0:
        return "-"
    return f"{value:.4f}"


def _render_table(headers: list[str], rows: list[list[str]], align_left: set[int] | None = None) -> list[str]:
    left = align_left or set()
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(cells: list[str]) -> str:
        padded = []
        for index, cell in enumerate(cells):
            text = cell.ljust(widths[index]) if index in left else cell.rjust(widths[index])
            padded.append(f" {text} ")
        return "|" + "|".join(padded) + "|"

    return [
        border(),
        render_row(headers),
        border(),
        *[render_row(row) for row in rows],
        border(),
    ]


def _capture_printed_tables() -> dict[str, Any]:
    """Keep the per-class rows the evaluator prints, so they can be saved."""
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback

    captured: dict[str, Any] = {"overall": None, "per_class": None, "split": None}
    original = COCOEvalCallback._print_metrics_tables

    def wrapped(self, trainer, split, overall, per_class, *args, **kwargs):
        original(self, trainer, split, overall, per_class, *args, **kwargs)
        rows = [dict(row) for row in (per_class or [])]
        if split not in {"val", "test"} or not isinstance(overall, dict):
            return
        if captured["per_class"] and not rows:
            return
        captured["split"] = split
        captured["overall"] = dict(overall)
        captured["per_class"] = rows

    COCOEvalCallback._print_metrics_tables = wrapped
    return captured


def _overall_from_metrics(metrics: dict[str, float], split: str) -> dict[str, float]:
    prefix = f"{split}/"
    mapping = {
        "mAP 50:95": f"{prefix}mAP_50_95",
        "mAP 50": f"{prefix}mAP_50",
        "mAP 75": f"{prefix}mAP_75",
        "mAR": f"{prefix}mAR",
        "F1": f"{prefix}F1",
        "Precision": f"{prefix}precision",
        "Recall": f"{prefix}recall",
        "segm mAP 50:95": f"{prefix}segm_mAP_50_95",
        "segm mAP 50": f"{prefix}segm_mAP_50",
    }
    return {
        label: metrics[key]
        for label, key in mapping.items()
        if key in metrics
    }


def _per_class_from_metrics(metrics: dict[str, float], split: str) -> list[dict[str, Any]]:
    prefix = f"{split}/AP/"
    rows = []
    for key, value in metrics.items():
        if key.startswith(prefix):
            rows.append({"name": key[len(prefix):], "ap": value})
    return sorted(rows, key=lambda row: str(row["name"]).lower())


def _report_lines(
    checkpoint: Path,
    dataset_dir: Path,
    model_size: str,
    resolution: int,
    split: str,
    overall: dict[str, float],
    per_class: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "=" * 60,
        " CHECKPOINT EVALUATION",
        "=" * 60,
        f"  When               : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  Checkpoint         : {checkpoint}",
        f"  Dataset            : {dataset_dir}",
        f"  Model              : RFDETRSeg{model_size.capitalize()}",
        f"  Resolution         : {resolution}",
        f"  Split              : {split}",
        "",
        "  Overall",
    ]
    if overall:
        headers = list(overall)
        values = [_fmt(overall[header]) for header in headers]
        lines.extend(f"  {row}" for row in _render_table(headers, [values]))
    else:
        lines.append("  No overall metrics were returned.")

    lines.append("")
    lines.append("  Per-class")
    if per_class:
        headers = ["Class", "AP 50:95", "AR", "F1", "Precision", "Recall"]
        rows = [
            [
                str(row.get("name", "")),
                _fmt(row.get("ap")),
                _fmt(row.get("ar")),
                _fmt(row.get("f1")),
                _fmt(row.get("precision")),
                _fmt(row.get("recall")),
            ]
            for row in per_class
        ]
        lines.extend(f"  {row}" for row in _render_table(headers, rows, align_left={0}))
    else:
        lines.append("  No per-class metrics were returned.")
    lines.append("=" * 60)
    return lines


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    resolution = args.resolution or DEFAULT_RESOLUTION[args.model_size]

    if not checkpoint.is_file():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not dataset_dir.is_dir():
        print(f"Error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1
    if args.batch_size < 1:
        print("Error: --batch-size must be >= 1.", file=sys.stderr)
        return 1

    report_path = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint.parent / "eval_report.txt"
    )

    print(f"Checkpoint   : {checkpoint}")
    print(f"Dataset      : {dataset_dir}")
    print(f"Model        : RFDETRSeg{args.model_size.capitalize()}")
    print(f"Resolution   : {resolution}")
    print(f"Split        : {args.split}")
    print(f"Batch size   : {args.batch_size}")

    model = MODEL_MAP[args.model_size].from_checkpoint(
        checkpoint,
        resolution=resolution,
        trust_checkpoint=True,
    )
    captured = _capture_printed_tables()
    metrics = model.evaluate(
        dataset_dir=str(dataset_dir),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        multi_scale=False,
        log_per_class_metrics=True,
        eval_base_model=True,
        output_dir=str(report_path.parent),
    )

    overall = captured["overall"] or _overall_from_metrics(metrics, args.split)
    per_class = captured["per_class"] or _per_class_from_metrics(metrics, args.split)
    lines = _report_lines(
        checkpoint,
        dataset_dir,
        args.model_size,
        resolution,
        args.split,
        overall,
        per_class,
    )
    report = "\n".join(lines) + "\n"
    print("\n" + report, flush=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    metrics_path = report_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report saved : {report_path}")
    print(f"Metrics saved: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
