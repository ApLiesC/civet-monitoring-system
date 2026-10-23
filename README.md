# civet-monitoring-system

This project prepares CCTV footage for civet detection and trains a YOLO11n object detection model to detect civets in real-world footage.

## Project Structure

```text
civet-monitoring-system/
├── data/
│   ├── videos/             # Original or manually trimmed CCTV videos
│   ├── raw_frames/         # Extracted frames
│   ├── selected_frames/    # Frames selected for annotation
│   ├── enhanced_frames/    # Enhanced images for annotation
│   └── civet_dataset/      # Exported YOLO dataset
├── scripts/
│   ├── extract_frames.py
│   ├── enhance_frames.py
│   └── train_detector.py
├── runs/
│   └── civet_yolo11n-2/    # Example training run output
├── requirements.txt
└── README.md
```

Actual output directories may differ depending on your configuration and previous runs.

## 1. Install Dependencies

Make sure Python is installed. Open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

## 2. Prepare CCTV Videos

Place the videos inside `data/videos/`.

- Organize videos into folders by behavior or camera if needed.
- Manually trim unnecessary sections where the civet is absent, too far away, or cannot be reliably identified.
- Preserve the source video names so extracted frames can be traced back to their original videos.

## 3. Extract Frames

Run the frame extraction script:

```bash
python scripts/extract_frames.py
```

The script extracts one frame every 2.5 seconds from each video in `data/videos/` and saves the images in `data/raw_frames/`.

Each filename includes its source video name and timestamp to prevent naming conflicts.

## 4. Select Frames

Review the extracted images and copy useful frames into `data/selected_frames/`.

Prioritize diversity rather than collecting many nearly identical frames. Include:

- Different body orientations and postures.
- Different eating poses, including the civet bending its head toward the food tray.
- Daytime and nighttime conditions.
- Different camera views, distances, and positions.
- Partial occlusions from bushes, trays, and enclosure objects.
- Difficult backgrounds where the civet blends into the ground.

Include images without civets when useful for teaching the model to distinguish the animal from background objects.

Keep track of each image's source video. Frames from the same video must remain in the same dataset split.

## 5. Enhance Selected Frames

Run the enhancement script:

```bash
python scripts/enhance_frames.py
```

The script applies grayscale conversion, mild Gaussian smoothing, CLAHE contrast enhancement, and gamma correction. Enhanced images are saved in `data/enhanced_frames/`, leaving the original images unchanged.

Compare enhanced images with their originals before processing the full dataset. Enhancement should preserve useful details around the civet's body outline, legs, snout, and fur pattern.

Use preprocessing consistently across training, validation, and test images. Real footage used for inference should follow the same preprocessing pipeline used to prepare the training images.

## 6. Annotate and Export the Dataset

Use Roboflow to annotate the images from `data/enhanced_frames/`.

1. Create one object-detection class named `civet`.
2. Draw a bounding box around each visible, identifiable civet.
3. Annotate the whole animal consistently rather than labeling separate body parts as individual civets.
4. For partially occluded animals, draw a consistent box around the identifiable visible extent of the animal.
5. Review the annotations, especially around food trays, shadows, and bushes.
6. Split the dataset into training, validation, and test sets.
7. Export the dataset in YOLO format into `data/civet_dataset/`.

**Important:** Split by source video rather than randomly splitting individual frames. Do not place frames from the same video in different splits, as similar frames can make evaluation results overly optimistic.

The exported dataset should contain a YOLO configuration file, typically `data.yaml`, and the corresponding images and labels.

## 7. Configure Model Training

Open `scripts/train_detector.py` and check the training configuration:

```python
MODEL_NAME = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 4
PATIENCE = 12
```

These are the current recommended starting settings for this project. The model may stop before reaching the maximum number of epochs if validation performance does not improve.

Check that the dataset path points to the exported YOLO configuration file:

```text
data/civet_dataset/data.yaml
```

The training script should use this dataset configuration and save its results to a run directory under `runs/`.

## 8. Train the Detector

From the project root, run:

```bash
python scripts/train_detector.py
```

Training uses the pretrained YOLO11n model and fine-tunes it on the annotated civet dataset.

After training, the run directory contains model weights and training results. The two main weight files are:

- `weights/best.pt` — model checkpoint with the best validation results.
- `weights/last.pt` — model checkpoint from the final training epoch.

For example, the latest run may save its best model at:

```text
runs/civet_yolo11n-2/weights/best.pt
```

Use the actual path printed by the training script or shown in the training log. The run directory may change when you train again.

## 9. Test the Model on Real CCTV Footage

Test the trained model on a CCTV video that was not used to train the model. Prefer footage containing challenging situations such as eating, partial occlusion, and low-contrast nighttime scenes.

### 9.1 Run inference

Open a terminal in the project root:

```bash
cd D:\civet-monitoring-system
```

Run the detector using the trained weights:

```bash
yolo detect predict model="models/civet_detector_yolo11n/weights/best.pt" source="video_file_path" conf=0.25 save=True
```

Replace the model path and video path with the actual locations of your files.

Parameters:

| Parameter | Purpose |
|---|---|
| `model` | Path to the trained `best.pt` weights |
| `source` | Path to the CCTV video to test |
| `conf=0.25` | Minimum detection confidence |
| `iou=0.45` | IoU threshold used for non-maximum suppression |
| `imgsz=640` | Inference image size |
| `save=True` | Saves the annotated output video |

The confidence and IoU values are starting points, not guaranteed optimal settings.

### 9.2 Find the output

Ultralytics saves inference results under a directory similar to:

```text
runs/detect/predict/
```

If the directory already exists, Ultralytics may create a numbered directory such as `predict2` or `predict3`. Check the terminal output for the exact save location.

Open the output video and inspect the detection boxes and confidence values throughout the clip.

### 9.3 What to check

Pay particular attention to:

- **Missed detections:** The civet is visible, but no box appears.
- **Duplicate detections:** Multiple boxes appear around the same civet.
- **False positives:** The model detects a food tray, bush, shadow, or other background object as a civet.
- **Eating poses:** Detection becomes unreliable when the civet bends its head, turns, or crouches near the tray.
- **Partial occlusion:** The model loses the civet when part of its body is hidden.
- **Nighttime footage:** Detection disappears when the civet's outline blends into the ground.

Record the approximate video timestamps of these failures so the corresponding frames can be reviewed and used to improve the dataset.

## 10. Improve the Model

If detection is unreliable, identify the failure type before changing model settings.

- Add diverse training examples for underrepresented poses and lighting conditions.
- Include useful background-only images containing trays, bushes, and shadows.
- Review bounding boxes for inconsistent or incorrect annotations.
- Compare enhancement settings on the same original frames before regenerating the dataset.
- Keep training, validation, and test data separated by source video.
- Retrain after updating the dataset, then test on held-out footage.

Avoid judging the model solely by confidence scores or validation mAP. A model can achieve strong validation metrics while still struggling with unfamiliar poses or real-world lighting.

## Notes

- Frame extraction, frame selection, enhancement, annotation, training, and inference are separate stages.
- Keep original CCTV footage unchanged.
- Apply the same preprocessing pipeline during training and real-footage inference.
- Do not use test images to tune the model repeatedly; reserve the test set for independent evaluation.
- Do not commit large CCTV videos, datasets, or trained weights to Git unless necessary and permitted.
