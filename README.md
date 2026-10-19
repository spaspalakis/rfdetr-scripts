# RF-DETR

Δύο ξεχωριστά trainings. Τρέξε κάθε εντολή μέσα στον φάκελο της.


```bash
pip install "rfdetr[train]" supervision
```

Κάθε `train.py` διαβάζει το `data.yaml` που είναι δίπλα του. Αν λείπει, σταματάει. Το checkpoint για inference είναι το `checkpoint_best_total.pth` μέσα στο run folder. Δίπλα του γράφονται `class_names.txt` και `training_summary.txt` (mAP συνολικά και ανά κλάση, στο validation set).

---

## detection

Το dataset είναι ήδη YOLO (`train/images`, `train/labels`). Δεν χρειάζεται μετατροπή σε COCO. Το path και οι κλάσεις είναι στο `detection/data.yaml`.

### `train.py`

```bash
python train.py \
  --model-size medium \
  --epochs 100 \
  --batch-size 2 \
  --grad-accum 4 \
  --skip-epochs 3
```

Default αναλύσεις: nano 384, small 512, medium 576, large 704. Αν τελειώσει η μνήμη της GPU, `--batch-size 1`.

Το run γράφεται στο `detection/output/<timestamp>_det_<size>_.../`. Στο τέλος τυπώνεται ο πίνακας mAP του καλύτερου checkpoint στο validation set.

### `evaluate.py`

Μετράει ξανά ένα αποθηκευμένο checkpoint. `--checkpoint` είναι υποχρεωτικό. Default dataset: το Reaction YOLO. Default split: `val`.

```bash
python evaluate.py \
  --checkpoint output/<run>/checkpoint_best_total.pth \
  --model-size medium \
  --split test
```

Η αναφορά πάει σε `eval_report.txt` δίπλα στο checkpoint. `--model-size` και `--resolution` πρέπει να είναι αυτά του training.

### `inference.py`

Όλες οι εικόνες ενός φακέλου, ή ένα βίντεο. Τα κουτιά ζωγραφίζονται πάντα. Τα ονόματα κλάσεων διαβάζονται από το `class_names.txt` δίπλα στο checkpoint.

Test εικόνες Reaction:

```bash
python inference.py \
  --model output/<run>/checkpoint_best_total.pth \
  --model-size medium \
  --folder /home/spaspalakis/Documents/REACTION/dataset-Reaction/Reaction.v9i.yolov8/test/images \
  --threshold 0.5
```

Βίντεο:

```bash
python inference.py \
  --model output/<run>/checkpoint_best_total.pth \
  --model-size medium \
  --video \
  --video-path /path/to/file.mp4 \
  --threshold 0.5
```

Οι annotated εικόνες γράφονται στο `detection/output/runs/<timestamp>_<όνομα>/`.

---

## segmentation

Το `segmentation/data.yaml` δείχνει σε COCO segmentation export: `train/`, `valid/` και προαιρετικά `test/`, το καθένα με `_annotations.coco.json`.

### `train.py`

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

Default αναλύσεις, αν δεν δώσεις `--resolution`: nano 312, small 384, medium 432, large 504. Η ανάλυση πρέπει να διαιρείται με το block size του μοντέλου.

Το run γράφεται στο `segmentation/output/<timestamp>_seg_<size>_.../`.

### `evaluate.py`

```bash
python evaluate.py \
  --checkpoint output/<run>/checkpoint_best_total.pth \
  --dataset-dir /path/to/coco-segmentation-dataset \
  --model-size medium \
  --split val
```

`--split test` σκοράρει το test split. Η αναφορά είναι `eval_report.txt` δίπλα στο checkpoint.

### `inference.py`

Χρειάζεται τουλάχιστον ένα από `--bbox` και `--polygon`. Τα ονόματα κλάσεων έρχονται από COCO JSON (`--annotations`).

Φάκελος εικόνων, κουτιά και μάσκες:

```bash
python inference.py \
  --model output/<run>/checkpoint_best_total.pth \
  --annotations /path/to/train/_annotations.coco.json \
  --model-size small \
  --folder /path/to/images \
  --bbox \
  --polygon \
  --threshold 0.5
```

Βίντεο:

```bash
python inference.py \
  --model output/<run>/checkpoint_best_total.pth \
  --annotations /path/to/train/_annotations.coco.json \
  --model-size small \
  --video \
  --video-path /path/to/file.mp4 \
  --bbox \
  --polygon
```

---

## Κοινά flags του `train.py`

Ίδια και στα δύο folders.

| Flag | Default | Τι κάνει |
|------|---------|----------|
| `--model-size` | `medium` | `nano`, `small`, `medium`, `large` |
| `--resolution` | ανά μέγεθος | Τετράγωνη είσοδος. Πρέπει να διαιρείται με το block size |
| `--epochs` | `200` | Μέγιστος αριθμός epochs |
| `--batch-size` | `4` | Batch ανά GPU |
| `--grad-accum` | `4` | Effective batch = `batch-size × grad-accum` |
| `--lr` | `1e-4` | Learning rate |
| `--early-stopping` / `--no-early-stopping` | on | Σταματάει όταν το validation mAP δεν ανεβαίνει |
| `--early-stopping-patience` | `10` | Epochs αναμονής |
| `--skip-epochs` | `0` | Τα πρώτα N epochs δεν μετράνε για best checkpoint. Για fine-tune βάλε `3` |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | on | Λιγότερη VRAM |
| `--tensorboard` / `--no-tensorboard` | on | TensorBoard logs |
| `--device` | `cuda` | `cuda`, `cpu`, `mps` |
| `--output` | `./output` | Βασικός φάκελος των runs |
| `--run-id` | auto | Όνομα φακέλου του run |
| `--resume` | — | Συνέχεια από `.ckpt` ή `.pth` |
