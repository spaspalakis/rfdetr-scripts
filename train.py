#!/usr/bin/env python3
"""RF-DETR segmentation training CLI."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from pytorch_lightning.callbacks import Callback
from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "data.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "output"

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


def _format_duration(total_seconds: float) -> str:
    total_minutes = total_seconds / 60
    hours, minutes = divmod(total_minutes, 60)
    if hours >= 1:
        return f"{int(hours)}h {minutes:.2f} min ({total_minutes:.2f} min)"
    return f"{total_minutes:.2f} min"


def _early_stopping_info(trainer) -> tuple[float | None, int | None]:
    for callback in trainer.callbacks:
        best_score = getattr(callback, "best_score", None)
        stopped_epoch = getattr(callback, "stopped_epoch", None)
        if best_score is not None or stopped_epoch is not None:
            return best_score, stopped_epoch
    return None, None


def _fmt_metric(value: float | None) -> str:
    if value is None or value != value or value < 0:
        return "-"
    return f"{value:.4f}"


def _callback_metric(trainer, key: str) -> float | None:
    value = trainer.callback_metrics.get(key)
    if value is None:
        return None
    number = float(value.item() if hasattr(value, "item") else value)
    if number != number:
        return None
    return number


def _render_table(headers: list[str], rows: list[list[str]], align_left: set[int] | None = None) -> list[str]:
    left = align_left or set()
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def border(left_char: str, mid_char: str, right_char: str, fill: str) -> str:
        return left_char + mid_char.join(fill * (width + 2) for width in widths) + right_char

    def render_row(cells: list[str]) -> str:
        padded = []
        for index, cell in enumerate(cells):
            text = cell.ljust(widths[index]) if index in left else cell.rjust(widths[index])
            padded.append(f" {text} ")
        return "|" + "|".join(padded) + "|"

    return [
        border("+", "+", "+", "-"),
        render_row(headers),
        border("+", "+", "+", "-"),
        *[render_row(row) for row in rows],
        border("+", "+", "+", "-"),
    ]


def _overall_table(overall: dict[str, float]) -> list[str]:
    mar_key = next((key for key in overall if key.startswith("mAR")), None)
    columns: list[tuple[str, float | None]] = [
        ("mAP 50:95", overall.get("mAP 50:95")),
        ("mAP 50", overall.get("mAP 50")),
        ("mAP 75", overall.get("mAP 75")),
        (mar_key or "mAR", overall.get(mar_key) if mar_key else None),
        ("F1", overall.get("F1")),
        ("Prec", overall.get("Precision")),
        ("Recall", overall.get("Recall")),
    ]
    if "segm mAP 50:95" in overall:
        columns.append(("segm 50:95", overall.get("segm mAP 50:95")))
        columns.append(("segm 50", overall.get("segm mAP 50")))
    headers = [label for label, _ in columns]
    values = [_fmt_metric(value) for _, value in columns]
    return _render_table(headers, [values])


def _per_class_table(per_class: list[dict]) -> list[str]:
    if not per_class:
        return []
    headers = ["Class", "AP 50:95", "AR", "F1", "Precision", "Recall"]
    rows = [
        [
            str(row.get("name", "")),
            _fmt_metric(row.get("ap")),
            _fmt_metric(row.get("ar")),
            _fmt_metric(row.get("f1")),
            _fmt_metric(row.get("precision")),
            _fmt_metric(row.get("recall")),
        ]
        for row in per_class
    ]
    return _render_table(headers, rows, align_left={0})


class TrainingSummaryCallback(Callback):
    def __init__(self, summary: dict):
        self.summary = summary
        self._fit_start: float | None = None
        self._epoch_start: float | None = None
        self._started_at: datetime | None = None
        self._val_history: dict[int, dict] = {}
        self._best_regular_map = float("-inf")
        self._best_regular_epoch: int | None = None
        self._best_ema_map = float("-inf")
        self._best_ema_epoch: int | None = None

    def on_fit_start(self, trainer, pl_module):
        self._fit_start = time.perf_counter()
        self._started_at = datetime.now()
        self._capture_validation_tables(trainer)

    def _capture_validation_tables(self, trainer) -> None:
        for callback in trainer.callbacks:
            printer = getattr(callback, "_print_metrics_tables", None)
            if not callable(printer):
                continue
            summary = self

            def wrapped(trainer, split, overall, per_class, _printer=printer):
                _printer(trainer, split, overall, per_class)
                if split == "val":
                    summary._record_validation(trainer, overall, per_class)

            callback._print_metrics_tables = wrapped
            return

    def _record_validation(self, trainer, overall: dict, per_class: list) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(trainer.current_epoch)
        ema_map = _callback_metric(trainer, "val/ema_mAP_50_95")
        self._val_history[epoch] = {
            "overall": dict(overall),
            "per_class": [dict(row) for row in per_class],
            "ema_map": ema_map,
            "ema_map_50": _callback_metric(trainer, "val/ema_mAP_50"),
            "ema_mar": _callback_metric(trainer, "val/ema_mAR"),
            "ema_segm_map": _callback_metric(trainer, "val/ema_segm_mAP_50_95"),
            "ema_segm_map_50": _callback_metric(trainer, "val/ema_segm_mAP_50"),
        }
        if epoch < int(self.summary.get("skip_epochs") or 0):
            return
        regular_map = float(overall.get("mAP 50:95", float("nan")))
        if regular_map == regular_map and regular_map > self._best_regular_map:
            self._best_regular_map = regular_map
            self._best_regular_epoch = epoch
        if ema_map is not None and ema_map > self._best_ema_map:
            self._best_ema_map = ema_map
            self._best_ema_epoch = epoch

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.perf_counter() - (self._epoch_start or time.perf_counter())
        print(
            f"[timing] Epoch {trainer.current_epoch} finished in {elapsed / 60:.2f} min",
            flush=True,
        )

    def on_fit_end(self, trainer, pl_module):
        total_seconds = time.perf_counter() - (self._fit_start or time.perf_counter())
        print(f"[timing] Total training time: {_format_duration(total_seconds)}", flush=True)

        completed_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        num_devices = max(trainer.num_devices, 1)
        effective_batch = (
            self.summary["batch_size"]
            * self.summary["grad_accum"]
            * num_devices
        )
        best_score, stopped_epoch = _early_stopping_info(trainer)
        early_stopped = completed_epoch < max_epochs

        lines = [
            "",
            "=" * 60,
            " TRAINING SUMMARY",
            "=" * 60,
            f"  Run ID              : {self.summary['run_id']}",
            f"  Started at          : {self._started_at:%Y-%m-%d %H:%M:%S}"
            if self._started_at
            else "  Started at          : n/a",
            f"  Finished at         : {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"  Total time          : {_format_duration(total_seconds)}",
            "",
            "  Dataset             : {dataset}".format(**self.summary),
            f"  Model               : RFDETRSeg{self.summary['model_size'].capitalize()}",
            f"  Resolution          : {self.summary['resolution']}",
            "",
            f"  Epochs completed    : {completed_epoch} / {max_epochs}",
            f"  Stopped early       : {'yes' if early_stopped else 'no'}",
            f"  Early stopping      : {'enabled' if self.summary['early_stopping'] else 'disabled'}",
            f"  Early-stop patience : {self.summary['early_stopping_patience']}",
            f"  Skip best epochs    : {self.summary['skip_epochs']}",
            "",
            f"  Batch size (per GPU): {self.summary['batch_size']}",
            f"  Grad accumulation   : {self.summary['grad_accum']}",
            f"  Effective batch     : {effective_batch} "
            f"({self.summary['batch_size']} x {self.summary['grad_accum']} x {num_devices})",
            f"  Learning rate       : {self.summary['lr']}",
        ]
        if self.summary.get("classes"):
            lines.append("  Classes             :")
            lines.extend(f"    {row}" for row in self.summary["classes"])
        if self.summary.get("resume"):
            lines.append(f"  Resumed from        : {self.summary['resume']}")
        if best_score is not None:
            lines.append(f"  Early-stop best mAP : {float(best_score):.4f}")
            lines.append("                        (patience score; ignores gains below min delta)")
        if stopped_epoch:
            lines.append(f"  Stopped during epoch: {int(stopped_epoch)}")

        lines.extend(self._best_model_lines())
        lines.extend(["=" * 60, ""])
        summary_text = "\n".join(lines)
        print(summary_text, flush=True)

        summary_path = Path(self.summary["output_dir"]) / "training_summary.txt"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(f"Summary saved to: {summary_path}", flush=True)

    def _best_model_lines(self) -> list[str]:
        has_regular = self._best_regular_epoch is not None
        has_ema = self._best_ema_epoch is not None
        if not has_regular and not has_ema:
            return ["", "  Best checkpoint     : n/a (no validation metrics recorded)"]

        regular_map = self._best_regular_map if has_regular else 0.0
        ema_map = self._best_ema_map if has_ema else 0.0
        best_is_ema = has_ema and ema_map > regular_map
        winner_epoch = self._best_ema_epoch if best_is_ema else self._best_regular_epoch
        winner_map = ema_map if best_is_ema else regular_map
        winner_source = "EMA" if best_is_ema else "regular"

        lines = [
            "",
            "  Best checkpoint     : "
            f"{winner_source} weights, epoch {winner_epoch}, mAP {winner_map:.4f}",
            "  Checkpoint file     : checkpoint_best_total.pth",
        ]
        if has_regular:
            lines.append(
                f"  Best regular mAP    : {self._best_regular_map:.4f} "
                f"(epoch {self._best_regular_epoch})"
            )
        if has_ema:
            lines.append(
                f"  Best EMA mAP        : {self._best_ema_map:.4f} "
                f"(epoch {self._best_ema_epoch})"
            )
        lines.append(
            "  Note                : mid-run 'Best regular/EMA mAP saved' lines are overwritten. "
            "This block is the final best."
        )
        lines.append(
            "  Note                : the mAP tables are the regular model. "
            "EMA mAP is logged separately and selects the checkpoint when it is higher."
        )

        if winner_epoch is not None:
            lines.extend(self._epoch_table_lines(
                winner_epoch,
                f"Best {winner_source} epoch {winner_epoch} — regular-model val table",
            ))
        if (
            has_regular
            and best_is_ema
            and self._best_regular_epoch != winner_epoch
        ):
            lines.extend(self._epoch_table_lines(
                self._best_regular_epoch,
                f"Best regular epoch {self._best_regular_epoch} — regular-model val table",
            ))
        return lines

    def _epoch_table_lines(self, epoch: int | None, title: str) -> list[str]:
        snapshot = self._val_history.get(epoch) if epoch is not None else None
        lines = ["", f"  {title}"]
        if snapshot is None:
            lines.append("  Metrics table was not captured for this epoch.")
            return lines
        if snapshot.get("ema_map") is not None:
            lines.append(
                "  EMA at this epoch   : "
                f"mAP 50:95 {_fmt_metric(snapshot.get('ema_map'))} | "
                f"mAP 50 {_fmt_metric(snapshot.get('ema_map_50'))} | "
                f"mAR {_fmt_metric(snapshot.get('ema_mar'))} | "
                f"segm 50:95 {_fmt_metric(snapshot.get('ema_segm_map'))} | "
                f"segm 50 {_fmt_metric(snapshot.get('ema_segm_map_50'))}"
            )
        lines.append("  Regular overall")
        lines.extend(f"  {row}" for row in _overall_table(snapshot["overall"]))
        per_class = _per_class_table(snapshot["per_class"])
        if per_class:
            lines.append("  Regular per-class")
            lines.extend(f"  {row}" for row in per_class)
        return lines


def _names_from_config(names: object, source: Path) -> list[str]:
    """Read a YOLO-style ``names`` list or ``{0: name}`` map."""
    if isinstance(names, list):
        ordered = [str(name).strip() for name in names]
    elif isinstance(names, dict):
        numeric_keys: list[int] = []
        for key in names:
            key_text = str(key)
            if not key_text.isdigit():
                raise ValueError(
                    f"{source} names must use integer keys 0..N-1, got {key!r}"
                )
            numeric_keys.append(int(key_text))
        if sorted(set(numeric_keys)) != list(range(len(numeric_keys))):
            raise ValueError(
                f"{source} names must be contiguous keys 0..N-1, got {sorted(set(numeric_keys))}"
            )
        ordered = []
        for index in range(len(numeric_keys)):
            value = names.get(index, names.get(str(index)))
            ordered.append(str(value).strip())
    else:
        raise ValueError(f"{source} field 'names' must be a list or a map of 0..N-1")
    if not ordered or any(not name for name in ordered):
        raise ValueError(f"{source} contains an empty class name")
    return ordered


def load_dataset_config(path: Path) -> dict[str, object]:
    """Load a YOLO-style dataset config with ``path`` and ``names``."""
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required to read dataset config files") from exc

    if not path.is_file():
        raise ValueError(f"dataset config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        try:
            payload = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping with 'path' and 'names'")

    dataset_path = payload.get("path", payload.get("dataset_dir"))
    names = payload.get("names")
    if names is None:
        raise ValueError(f"{path} must define 'names', the class list in model order")
    return {
        "path": Path(str(dataset_path)).expanduser() if dataset_path else None,
        "names": _names_from_config(names, path),
        "source": path,
    }


def resolve_training_dataset() -> tuple[Path, Path, list[str]]:
    """Load ``data.yaml`` next to this script and return its dataset path and classes."""
    if not DATA_YAML.is_file():
        raise ValueError(
            f"{DATA_YAML} was not found. Create it with 'path' and 'names'."
        )

    dataset_config = load_dataset_config(DATA_YAML)
    dataset_path = dataset_config.get("path")
    if dataset_path is None:
        raise ValueError(f"{DATA_YAML} must define 'path'")

    resolved_dir = resolve_path(Path(str(dataset_path)))
    if not resolved_dir.is_dir():
        raise ValueError(f"dataset directory not found: {resolved_dir}")

    names = dataset_config["names"]
    if not isinstance(names, list):
        raise ValueError(f"{DATA_YAML} has no class names")
    return resolved_dir, DATA_YAML, names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RF-DETR segmentation model on a COCO/YOLO dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Base output directory for training runs (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_MAP),
        default="medium",
        help="Model size: nano, small, medium, or large (default: medium).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Input resolution (must be divisible by model block size). "
        "Defaults: nano=312, small=384, medium=432, large=504.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-GPU batch size (default: 4).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4). "
        "Effective batch = batch_size x grad_accum x num_gpus.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume training (.ckpt or .pth).",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when validation mAP stops improving (default: enabled).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10).",
    )
    parser.add_argument(
        "--skip-epochs",
        type=int,
        default=0,
        help="Ignore the first N epochs for best-checkpoint selection and early "
        "stopping (default: 0). Use 3 when fine-tuning from pretrained weights.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade compute for lower VRAM usage (default: enabled).",
    )
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TensorBoard logging (default: enabled).",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "mps"),
        default="cuda",
        help="Training device (default: cuda).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional custom run folder name. Auto-generated if omitted.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.expanduser().is_absolute():
        candidates = [path.expanduser()]
        if len(path.parts) > 1:
            relative_tail = Path(*path.parts[1:])
            candidates.extend((Path.cwd() / relative_tail, SCRIPT_DIR / relative_tail))
    else:
        candidates = [Path.cwd() / path, SCRIPT_DIR / path]

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return path.expanduser().resolve()


def build_run_id(args: argparse.Namespace, resolution: int) -> str:
    if args.run_id:
        return args.run_id
    now = datetime.now()
    return (
        f"{now:%y-%m-%d}_{now:%H}:{now:%M}_seg_{args.model_size}_"
        f"r{resolution}_b{args.batch_size}_ga{args.grad_accum}_e{args.epochs}"
    )


def main() -> int:
    args = parse_args()

    if args.grad_accum < 1:
        print("Error: --grad-accum must be >= 1.", file=sys.stderr)
        return 1

    if args.skip_epochs < 0:
        print("Error: --skip-epochs must be >= 0.", file=sys.stderr)
        return 1

    try:
        dataset_dir, config_path, class_names = resolve_training_dataset()
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    class_lines = [f"{index}  {name}" for index, name in enumerate(class_names)]
    class_origin = f"{len(class_names)} from {config_path}"

    resolution = args.resolution or DEFAULT_RESOLUTION[args.model_size]
    output_base = resolve_path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    run_id = build_run_id(args, resolution)
    output_dir = output_base / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = str(resolve_path(args.resume)) if args.resume else None
    if args.resume and not Path(resume).is_file():
        print(f"Error: resume checkpoint not found: {resume}", file=sys.stderr)
        return 1

    effective_batch = args.batch_size * args.grad_accum
    (output_dir / "class_names.txt").write_text(
        "\n".join(class_names) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset dir      : {dataset_dir}", flush=True)
    if config_path is not None:
        print(f"Dataset config   : {config_path}", flush=True)
    print(f"Classes          : {class_origin}", flush=True)
    for line in class_lines:
        print(f"  {line}", flush=True)
    print(f"Model            : RFDETRSeg{args.model_size.capitalize()}")
    print(f"Resolution       : {resolution}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch / accum    : {args.batch_size} x {args.grad_accum} = {effective_batch} effective")
    print(f"Learning rate    : {args.lr}")
    print(f"Skip epochs      : {args.skip_epochs}")
    print(f"Output dir       : {output_dir}")
    if resume:
        print(f"Resume from      : {resume}")

    model_cls = MODEL_MAP[args.model_size]
    # resolution is a ModelConfig field — must be set at init, not on TrainConfig.
    model = model_cls(resolution=resolution)

    config = model.get_train_config(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        device=args.device,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        skip_best_epochs=args.skip_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        tensorboard=args.tensorboard,
        resume=resume,
        class_names=class_names,
    )

    module = RFDETRModelModule(model.model_config, config)
    datamodule = RFDETRDataModule(model.model_config, config)
    trainer = build_trainer(config, model.model_config)

    trainer.callbacks.append(
        TrainingSummaryCallback(
            {
                "run_id": run_id,
                "dataset": str(dataset_dir),
                "classes": class_lines,
                "model_size": args.model_size,
                "resolution": resolution,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "early_stopping": args.early_stopping,
                "early_stopping_patience": args.early_stopping_patience,
                "skip_epochs": args.skip_epochs,
                "output_dir": str(output_dir),
                "resume": resume,
            }
        )
    )

    summary_callback = trainer.callbacks[-1]
    trainer.fit(module, datamodule, ckpt_path=resume)

    model.model.model = module.model
    best_ckpt = output_dir / "checkpoint_best_total.pth"
    print(f"Done. Best model for inference: {best_ckpt}")
    if isinstance(summary_callback, TrainingSummaryCallback):
        _validate_best_checkpoint(trainer, module, datamodule, best_ckpt, summary_callback)
    return 0


def _validate_best_checkpoint(trainer, module, datamodule, best_ckpt: Path, summary_callback) -> None:
    """Measure the saved checkpoint once more and print its validation table.

    The tables printed during training are the regular weights of each epoch.
    This pass loads ``checkpoint_best_total.pth`` (the EMA weights when EMA won)
    and runs the validation set on those exact weights.
    """
    if not best_ckpt.is_file():
        print(f"Final validation skipped, checkpoint not found: {best_ckpt}", flush=True)
        return

    import torch

    checkpoint = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    wrapped = getattr(module.model, "_orig_mod", None)
    raw_model = wrapped if isinstance(wrapped, torch.nn.Module) else module.model
    raw_model.load_state_dict(checkpoint["model"], strict=True)

    for callback in trainer.callbacks:
        name = type(callback).__name__
        if name in {"BestModelCallback", "RFDETREarlyStopping", "ModelCheckpoint"}:
            callback.on_validation_end = lambda *args, **kwargs: None
        if callable(getattr(callback, "get_ema_model_state_dict", None)):
            callback._average_model = None
        if hasattr(callback, "map_metric_ema"):
            callback.map_metric_ema = None

    print(
        "\n"
        + "=" * 60
        + "\n BEST CHECKPOINT VALIDATION\n"
        + "=" * 60
        + f"\n  File                : {best_ckpt}\n"
        + "  These tables are this file, measured on the validation set.\n"
        + "  Use them in a presentation. mAP 50:95 is the main score.\n"
        + "  Per-class AP 50:95 is the score of each class.\n"
        + "  segm mAP is the mask score. F1 is a separate threshold sweep.\n"
        + "=" * 60,
        flush=True,
    )
    trainer.validate(module, datamodule=datamodule)

    snapshot = summary_callback._val_history.get(int(trainer.current_epoch))
    if not snapshot:
        return
    lines = [
        "",
        "=" * 60,
        " BEST CHECKPOINT VALIDATION",
        "=" * 60,
        f"  File                : {best_ckpt}",
        "  Split               : validation",
    ]
    lines.extend(summary_callback._epoch_table_lines(
        int(trainer.current_epoch),
        "Saved checkpoint — validation table",
    ))
    lines.append("=" * 60)
    report = "\n".join(lines) + "\n"
    print(report, flush=True)
    summary_path = best_ckpt.parent / "training_summary.txt"
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(report)
    print(f"Final validation appended to: {summary_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
