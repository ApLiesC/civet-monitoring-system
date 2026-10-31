# Civet classification

Place this folder at `scripts/classification/` in the civet-monitoring-system project. Run commands below from the project root. Uses the existing `ultralytics` dependency; install with `python -m pip install -r requirements.txt`.

## Prepare labeled images

The existing cropping script writes unlabeled crops to `data/behavior_crops/unlabeled`. Manually assign each crop one class, then organize your classification dataset using your two labels, `resting` and `eating`:

```text
data/civet_classification/
  train/
    eating/
    resting/
  val/
    eating/
    resting/
  test/
    eating/
    resting/
```

Put images inside every class folder. All splits must have the same class folders, with at least two classes. YOLO detection bounding-box labels and `data.yaml` are not classification labels. No automatic behavior labels are inferred from detections.

Keep all frames from the same source video or recording session in one split to avoid leakage. Use train for learning, val for choosing settings, and test only for final evaluation. Match crop padding, color/enhancement, and framing across splits and inference. A single-frame classifier may not distinguish behaviors that require motion over time.

## Train

```powershell
python scripts/classification/train_classifier.py
```

Optional settings: `--device 0 --epochs 100 --batch 16` for an NVIDIA GPU, or `--device cpu`. The initial pretrained `yolo11n-cls.pt` download requires internet access. Use `--model path/to/local-classifier.pt` for local weights. Workers default to 0 for Windows compatibility.

Training prints the actual best weights path. Repeated runs receive numbered output folders; use the printed path in subsequent commands.

## Test on labeled held-out images

```powershell
python scripts/classification/test_classification.py --model models/civet_classifier_yolo11n/weights/best.pt
```

Explicitly evaluates `test/`, without falling back to validation data. Saves accuracy to `evaluation.json` and Ultralytics plots under `runs/classify/test` (or a numbered folder). `--split val` evaluates validation data instead. Top-5 accuracy is uninformative for five or fewer classes; focus on top-1 accuracy and the confusion matrix.

## Predict images or video

```powershell
python scripts/classification/test_classification.py --model models/civet_classifier_yolo11n/weights/best.pt --source data/behavior_crops/unlabeled
python scripts/classification/test_classification.py --model models/civet_classifier_yolo11n/weights/best.pt --source path/to/cropped_civet_video.mp4
```

Prints a predicted class and confidence per image/frame and saves annotated outputs under `runs/classify/predict` (or a numbered folder). Add `--show` for a display window. Video results are streamed to avoid keeping all frames in memory. Prediction is not an accuracy test without ground-truth labels.

Classification assigns one label to the entire input. For a classifier trained on animal crops, use matching crops at prediction time. This command does not automatically detect or crop civets in full CCTV frames.

All commands support `--help`, `--imgsz`, `--device`, `--project`, and `--name`. Training and evaluation support `--data` for a custom dataset root. Use the same image size for training and testing.

API references: https://docs.ultralytics.com/tasks/classify/ and https://docs.ultralytics.com/models/yolo11/.

