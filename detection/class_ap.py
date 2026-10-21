"""Per-class AP50 and AP50:95 for an RF-DETR checkpoint on a Roboflow COCO split.

RF-DETR prints per-class AP 50:95 only. This runs the model over one split and
scores it with RF-DETR's own COCO metric twice: once at IoU 0.50 (AP50) and
once over IoU 0.50:0.95. Used by train_custom.py and evaluate.py.
"""

from __future__ import annotations

import json
from pathlib import Path

COCO_JSON = "_annotations.coco.json"
SPLIT_DIRS = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def split_annotations(dataset_dir: Path, split: str) -> Path:
    path = Path(dataset_dir) / SPLIT_DIRS.get(split, split) / COCO_JSON
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return path


def per_class_ap(model, dataset_dir: Path, split: str, class_names: list[str], score_threshold: float = 0.001) -> dict:
    """Return {"overall": {...}, "per_class": [{name, ap50, ap50_95, boxes}], "images": n}."""
    import torch
    from PIL import Image
    from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision

    ann_file = split_annotations(dataset_dir, split)
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    index_of = {name: i for i, name in enumerate(class_names)}
    cat_index = {
        cat["id"]: index_of[cat["name"]] for cat in data.get("categories", []) if cat.get("name") in index_of
    }
    boxes_by_image: dict[int, list] = {}
    for ann in data.get("annotations", []):
        label = cat_index.get(ann["category_id"])
        if label is None:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        boxes_by_image.setdefault(ann["image_id"], []).append(([x, y, x + w, y + h], label))

    metrics = {
        "ap50": OnePassCocoMeanAveragePrecision(
            iou_thresholds=[0.5], class_metrics=True, max_detection_thresholds=[1, 10, 500]
        ),
        "ap50_95": OnePassCocoMeanAveragePrecision(class_metrics=True, max_detection_thresholds=[1, 10, 500]),
    }
    images = data.get("images", [])
    for image in images:
        path = ann_file.parent / image["file_name"]
        detections = model.predict(Image.open(path).convert("RGB"), threshold=score_threshold)
        pred = {
            "boxes": torch.as_tensor(detections.xyxy, dtype=torch.float32).reshape(-1, 4),
            "scores": torch.as_tensor(detections.confidence, dtype=torch.float32).reshape(-1),
            "labels": torch.as_tensor(detections.class_id, dtype=torch.int64).reshape(-1),
        }
        gt = boxes_by_image.get(image["id"], [])
        target = {
            "boxes": torch.tensor([b for b, _ in gt], dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([label for _, label in gt], dtype=torch.int64),
        }
        for metric in metrics.values():
            metric.update([pred], [target])

    results = {key: metric.compute() for key, metric in metrics.items()}
    counts = [0] * len(class_names)
    for gt in boxes_by_image.values():
        for _, label in gt:
            counts[label] += 1

    def per_class_values(result) -> dict[int, float]:
        classes = [int(c) for c in result["classes"].reshape(-1).tolist()]
        values = [float(v) for v in result["map_per_class"].reshape(-1).tolist()]
        return dict(zip(classes, values))

    ap50 = per_class_values(results["ap50"])
    ap50_95 = per_class_values(results["ap50_95"])
    rows = [
        {
            "name": name,
            "ap50": ap50.get(index, -1.0),
            "ap50_95": ap50_95.get(index, -1.0),
            "boxes": counts[index],
        }
        for index, name in enumerate(class_names)
    ]
    return {
        "split": split,
        "images": len(images),
        "overall": {"mAP 50": float(results["ap50"]["map"]), "mAP 50:95": float(results["ap50_95"]["map"])},
        "per_class": rows,
    }


def format_table(report: dict) -> list[str]:
    """Plain text table, same style as the other reports."""

    def fmt(value: float) -> str:
        return "-" if value is None or value < 0 else f"{value:.4f}"

    headers = ["Class", "AP50", "AP 50:95", "Boxes"]
    rows = [[r["name"], fmt(r["ap50"]), fmt(r["ap50_95"]), str(r["boxes"])] for r in report["per_class"]]
    rows.append(["all", fmt(report["overall"]["mAP 50"]), fmt(report["overall"]["mAP 50:95"]), str(sum(r["boxes"] for r in report["per_class"]))])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells):
        return "|" + "|".join(f" {c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])} " for i, c in enumerate(cells)) + "|"

    return [
        f"Per-class AP on {report['split']} ({report['images']} images)",
        border,
        line(headers),
        border,
        *[line(row) for row in rows[:-1]],
        border,
        line(rows[-1]),
        border,
    ]
