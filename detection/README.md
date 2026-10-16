# RF-DETR object detection

`train.py` trains an RF-DETR detection model (boxes). The dataset path and class names live in `data.yaml` next to the script.

The Reaction export is already YOLO detection (`train/images`, `train/labels`, `data.yaml` inside the dataset). RF-DETR reads that layout directly. No COCO conversion and no copy of the images.

Classes, in model order: `boat`, `car`, `human`.

## Train

From this folder, with the RF-DETR environment active:

```bash
cd /home/spaspalakis/Documents/rf-detr/detection
python train.py \
  --model-size medium \
  --epochs 100 \
  --batch-size 2 \
  --grad-accum 4 \
  --skip-epochs 3
```

Resolution defaults: nano 384, small 512, medium 576, large 704. Effective batch is `batch-size × grad-accum`.

If the GPU runs out of memory, rerun with `--batch-size 1`.

## Output

Each run writes `output/<timestamp>_det_<size>_r<resolution>_.../`.

Use `checkpoint_best_total.pth` in that folder. The same folder has `class_names.txt` and `training_summary.txt`.

## Evaluate

```bash
python evaluate.py \
  --checkpoint output/<run>/checkpoint_best_total.pth \
  --model-size medium \
  --split val
```

`--split test` scores the held-out test images.

## Inference

```bash
python inference.py \
  --checkpoint output/<run>/checkpoint_best_total.pth \
  --model-size medium \
  --folder /path/to/images \
  --threshold 0.5
```

For a video, use `--video --video-path /path/to/file.mp4` instead of `--folder`.
