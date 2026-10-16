# RF-DETR training

`train.py` trains an RF-DETR segmentation model. The dataset path and the class names live in `data.yaml`, next to the script. Resolution, batch size, and the other training settings are command-line arguments.

## data.yaml

Create `data.yaml` in the same folder as `train.py`:

```yaml
path: /path/to/dataset
names:
  0: class_a
  1: class_b
  2: class_c
```

`path` is the dataset root. It must contain `train/` and `valid/`, each with images and a `_annotations.coco.json` file (a Roboflow COCO segmentation export).

`names` is the class list. Index `0` is the first model class. The keys must be `0, 1, 2, ...` with no gaps. On another machine, change `path` only. Keep the same names in the same order.

At start, training prints the path and the classes it loaded from this file. If `data.yaml` is missing, training stops.

## Train

```bash
python train.py \
  --model-size small \
  --resolution 768 \
  --batch-size 1 \
  --grad-accum 8 \
  --epochs 200 \
  --early-stopping-patience 10 \
  --skip-epochs 3
```

Effective batch size is `batch-size × grad-accum`. The example above is `1 × 8 = 8`.

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model-size` | `medium` | `nano`, `small`, `medium`, or `large` |
| `--resolution` | size default | Input resolution. Defaults: nano 312, small 384, medium 432, large 504 |
| `--epochs` | `200` | Maximum epochs |
| `--batch-size` | `4` | Batch size per GPU |
| `--grad-accum` | `4` | Gradient accumulation steps |
| `--lr` | `1e-4` | Learning rate |
| `--early-stopping` / `--no-early-stopping` | enabled | Stop when validation mAP stops improving |
| `--early-stopping-patience` | `10` | Epochs to wait before stopping |
| `--skip-epochs` | `0` | Ignore the first N epochs for the best checkpoint and early stopping. Use `3` when fine-tuning |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | enabled | Lower VRAM use |
| `--tensorboard` / `--no-tensorboard` | enabled | TensorBoard logs |
| `--device` | `cuda` | `cuda`, `cpu`, or `mps` |
| `--output` | `./output` | Base folder for runs |
| `--run-id` | auto | Custom run folder name |
| `--resume` | none | Checkpoint to resume from (`.ckpt` or `.pth`) |

## Output

Each run writes a folder under `output/`. The name looks like:

```text
26-09-30_11:35_seg_small_r768_b1_ga8_e200
```

The checkpoint to use for inference is `checkpoint_best_total.pth` inside that folder. The same folder also gets `class_names.txt` and `training_summary.txt`.
