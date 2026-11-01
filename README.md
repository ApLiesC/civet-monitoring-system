# Civet Monitoring System

A computer vision project for monitoring civet activity using CCTV footage. The system uses YOLO-based object detection and behavior classification to identify civets, determine their activity states, and record behavior events with timestamps.

## Project Overview

The project consists of four main stages:

1. **Data Preparation** – Extract, select, and enhance CCTV frames for dataset creation.
2. **Civet Detection** – Train a YOLO11n object detection model to locate civets in CCTV footage.
3. **Behavior Classification** – Train a classification model to recognize civet behaviors.
4. **Behavior Monitoring Pipeline** – Combine detection, movement analysis, and classification to generate activity records and behavior event summaries.

## Project Structure

```text
civet-monitoring-system/
├── configs/
│   ├── config.json                       # Image enhancement settings
│   └── behavior_pipeline_config.json    # Behavior pipeline settings
│
├── data/
│   ├── videos/                            # Original or trimmed CCTV videos
│   ├── raw_frames/                        # Extracted frames
│   ├── selected_frames/                   # Frames selected for annotation
│   ├── enhanced_frames/                   # Enhanced images
│   └── civet_dataset/                     # Exported YOLO detection dataset
│
├── models/
│   ├── civet_detector_yolo11n/            # Trained civet detector
│   └── civet_classifier_yolo11n/          # Trained behavior classifier
│
├── scripts/
│   ├── behavior_pipeline/                 # Complete behavior monitoring pipeline
│   ├── classification/                    # Behavior classifier training and testing
│   ├── detection/                         # Civet detector training and testing
│   └── preprocessing/                     # Frame extraction and image enhancement
│
├── runs/                                  # Training and inference outputs
├── outputs/
│   └── behavior_pipeline/                 # Behavior pipeline results
│
├── requirements.txt
└── README.md
```

Actual output directories and model paths may differ depending on configuration and previous runs.

## 1. Installation

Make sure Python is installed. Open a terminal in the project root and install the dependencies:

```bash
pip install -r requirements.txt
```

## 2. Prepare CCTV Videos

Place the original CCTV videos inside:

```text
data/videos/
```

Recommended preparation:

- Organize videos by camera, date, or behavior if needed.
- Manually trim unnecessary sections where the civet is absent, too far away, or cannot be reliably identified.
- Preserve the original video files so extracted frames can be traced back to their sources.
- Use footage from different lighting conditions, camera views, and civet behaviors.

Keep the original CCTV footage unchanged.

## 3. Extract and Select Frames

Use the frame extraction scripts in:

```text
scripts/preprocessing/
```

Extracted frames are saved in:

```text
data/raw_frames/
```

Review the extracted frames and copy useful images into:

```text
data/selected_frames/
```

Prioritize diversity rather than collecting many nearly identical frames. Include:

- Different body orientations and postures.
- Eating poses, including the civet bending toward the food tray.
- Daytime and nighttime conditions.
- Different camera views, distances, and positions.
- Partial occlusions from bushes, trays, and enclosure objects.
- Difficult backgrounds where the civet blends into the ground.

Include background-only images when useful for improving detection.

Frames from the same source video should remain in the same dataset split.

## 4. Enhance Selected Frames

Image enhancement settings are stored in:

```text
configs/config.json
```

The preprocessing scripts apply image enhancement operations such as:

- Grayscale conversion.
- Gaussian smoothing.
- CLAHE contrast enhancement.
- Gamma correction.

Enhanced images are saved in:

```text
data/enhanced_frames/
```

Original images remain unchanged.

Compare enhanced images with their originals before processing the full dataset. Make sure enhancement preserves useful details around the civet's body outline, legs, snout, and fur pattern.

Use consistent preprocessing for training, validation, and test images. CCTV footage used for inference should also follow the preprocessing approach used when preparing the training images.

## 5. Annotate and Export the Detection Dataset

Use Roboflow to annotate images from `data/enhanced_frames/`.

1. Create one object detection class named `civet`.
2. Draw a bounding box around each visible, identifiable civet.
3. Annotate the whole animal consistently rather than labeling separate body parts.
4. For partially occluded animals, draw a consistent box around the identifiable visible extent.
5. Review annotations around food trays, shadows, and bushes.
6. Split the dataset into training, validation, and test sets.
7. Export the dataset in YOLO format into `data/civet_dataset/`.

**Important:** Split the dataset by source video rather than randomly splitting individual frames. Frames from the same video should not appear in different splits, as similar frames can make evaluation results overly optimistic.

The exported dataset should contain a YOLO configuration file, typically `data.yaml`, along with the corresponding images and labels.

## 6. Train the Civet Detection Model

The detection scripts are located in:

```text
scripts/detection/
```

The project uses a pretrained YOLO11n model fine-tuned on the civet detection dataset.

The current training configuration uses the following starting settings:

```python
MODEL_NAME = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 4
PATIENCE = 12
```

The dataset configuration should point to:

```text
data/civet_dataset/data.yaml
```

Run the detector training script from the project root:

```bash
python scripts/detection/train_detector.py
```

After training, the output directory contains model weights and training results.

- `best.pt` – Model checkpoint with the best validation performance.
- `last.pt` – Model checkpoint from the final training epoch.

For example:

```text
models/civet_detector_yolo11n/weights/best.pt
```

Use the actual model path produced by your training process.

## 7. Train the Behavior Classification Model

The classification scripts are located in:

```text
scripts/classification/
```

The behavior classifier is used to recognize the civet's behavior from detected civet image crops.

The classification dataset should contain labeled examples of the behaviors supported by the model.

The current pipeline expects the classifier to distinguish:

- **Eating** – The civet is eating.
- **Resting** – The civet is stationary or resting.

The pipeline also maps resting-like labels, such as sitting, lying, and sleeping, to the general `resting` behavior.

Train and evaluate the classifier using the scripts in `scripts/classification/`. The trained classifier weights are used by the behavior monitoring pipeline.

Example model path:

```text
models/civet_classifier_yolo11n/weights/best.pt
```

## 8. Test the Models on CCTV Footage

Test both models using footage that was not used during training.

For the detection model, inspect:

- Missed detections.
- Duplicate bounding boxes around the same civet.
- False positives from trays, bushes, and shadows.
- Partial occlusion.
- Daytime and nighttime performance.

For the classification model, inspect:

- Incorrect behavior predictions.
- Confusion between eating and resting.
- Predictions with low confidence.
- Performance on different civet positions and camera views.

Use unfamiliar footage whenever possible to evaluate real-world performance.

## 9. Behavior Monitoring Pipeline

The behavior monitoring pipeline combines the civet detector, movement analyzer, and behavior classifier to process CCTV footage and generate activity and behavior records.

### 9.1 Pipeline Overview

```text
CCTV Video
    |
    v
Frame Sampling
    |
    v
Civet Detection (YOLO11n)
    |
    v
Civet Bounding Box and Crop
    |
    +--------------------------+
    |                          |
    v                          v
Movement Analysis        Behavior Classification
    |                          |
    v                          v
Locomotion /              Eating / Resting
Stationary / Unknown           |
    |                          |
    +-------------+------------+
                  |
                  v
           Activity Analysis
                  |
                  v
        Activity State Tracking
                  |
          +-------+-------+
          |               |
          v               v
        Active          Inactive
          |               |
          +-------+-------+
                  |
                  v
          Event Tracking
                  |
                  v
           Output Records
```

### 9.2 Activity States

The pipeline identifies three overall activity states:

| State | Description |
|---|---|
| `active` | The civet is moving (locomotion) or eating. |
| `inactive` | The civet is detected, stationary, and classified as resting. |
| `not_visible` | The detector cannot locate the civet for longer than the configured tolerance. |

Activity transitions are confirmed over a configurable duration to reduce rapid state changes caused by inconsistent predictions.

Unknown predictions are handled separately rather than automatically being classified as inactive.

### 9.3 Movement Analysis

The movement analyzer uses changes in the center of the civet's bounding box to estimate movement speed.

- `locomotion` – Movement speed reaches or exceeds the configured threshold.
- `stationary` – Movement speed remains below the threshold.
- `unknown` – There is insufficient movement history to determine the state.

Movement analysis helps distinguish locomotion from stationary behaviors such as eating and resting.

### 9.4 Behavior Classification

When the civet is detected and classified as stationary, its image crop is passed to the behavior classifier.

The current classifier recognizes:

- `eating`
- `resting`

Locomotion is determined through movement analysis rather than the behavior classifier.

Predictions below the configured confidence threshold are treated as uncertain.

### 9.5 Eating Event Tracking

The pipeline records eating episodes separately from overall activity states.

Each eating event includes:

- Event ID.
- Start time.
- End time.
- Duration.

The event tracker uses a minimum duration and gap tolerance to avoid recording very short or interrupted predictions as separate eating events.

The current configuration uses:

- Minimum eating duration: 5 seconds.
- Gap tolerance: 3 seconds.

The duration includes tolerated interruptions within an eating episode.

### 9.6 Pipeline Outputs

The pipeline generates an annotated video and several CSV files.

| Output | Description |
|---|---|
| `annotated_video.mp4` | CCTV footage with detection boxes, behavior labels, and movement information. |
| `behavior_results.csv` | Frame-level detection, movement, classification, and activity information. |
| `activity_records.csv` | Confirmed activity intervals, including active and inactive periods. |
| `eating_records.csv` | Recorded eating episodes with start time, end time, and duration. |
| `eating_frequency.csv` | Number of eating episodes within each configured time interval. |

The annotated video displays labels such as:

- `ACTIVE` – The civet is moving.
- `ACTIVE - Eating` – The civet is eating.
- `INACTIVE - Resting` – The civet is resting.

The CSV records provide more detailed timestamps and information for later analysis.

## 10. Configure and Run the Behavior Pipeline

Open:

```text
configs/behavior_pipeline_config.json
```

This file controls the models, input video, detection settings, classification settings, movement threshold, activity transition settings, eating event settings, and output configuration.

Example configuration:

```json
{
    "models": {
        "detector": "models/civet_detector_yolo11n-2/weights/best.pt",
        "classifier": "models/civet_classifier_yolo11n/weights/best.pt"
    },
    "video": {
        "input": "data/videos/locomotion11.mp4",
        "sample_fps": 8
    },
    "detection": {
        "confidence": 0.25,
        "image_size": 640
    },
    "classification": {
        "image_size": 224,
        "minimum_confidence": 0.70
    },
    "movement": {
        "movement_threshold": 70.0,
        "movement_window": 3
    },
    "activity": {
        "transition_duration": 3.0,
        "unknown_tolerance": 3.0,
        "not_visible_tolerance": 3.0
    },
    "eating": {
        "minimum_duration": 5.0,
        "gap_tolerance": 3.0
    },
    "cropping": {
        "padding_ratio": 0.10
    },
    "output": {
        "directory": "outputs/behavior_pipeline",
        "video_filename": "annotated_video.mp4",
        "csv_filename": "behavior_results.csv",
        "activity_csv_filename": "activity_records.csv",
        "eating_csv_filename": "eating_records.csv",
        "frequency_csv_filename": "eating_frequency.csv",
        "frequency_interval_seconds": 3600
    }
}
```

Update the model paths and input video path to match your files before running the pipeline.

Run the pipeline from the project root:

```bash
python scripts/behavior_pipeline/main.py
```

### Configuration Reference

| Setting | Description |
|---|---|
| `sample_fps` | Number of video frames processed per second. |
| `detection.confidence` | Minimum confidence for civet detection. |
| `classification.minimum_confidence` | Minimum confidence for accepting a behavior prediction. |
| `movement_threshold` | Movement speed threshold in pixels per second. |
| `movement_window` | Number of sampled positions used for movement analysis. |
| `transition_duration` | Time required to confirm an activity state transition. |
| `unknown_tolerance` | Maximum tolerated duration of uncertain classification. |
| `not_visible_tolerance` | Time before the civet is considered not visible. |
| `minimum_duration` | Minimum duration required to confirm an eating event. |
| `gap_tolerance` | Maximum interruption tolerated within an eating event. |
| `frequency_interval_seconds` | Time interval used to count eating events. `3600` represents one hour. |
| `cropping.padding_ratio` | Additional padding around the detected civet crop. |

### Output Location

By default, pipeline results are saved in:

```text
outputs/behavior_pipeline/
```

The directory contains the annotated video and CSV records. Output filenames and directory can be changed in `behavior_pipeline_config.json`.

## 11. Improve Model Performance

If the system produces unreliable results, identify the failure type before changing model settings.

**Detection issues**
- Add diverse examples for underrepresented poses and lighting conditions.
- Include background-only images containing trays, bushes, and shadows.
- Review bounding boxes for inconsistent or incorrect annotations.
- Test on unfamiliar CCTV footage.

**Classification issues**
- Add more examples of difficult eating and resting poses.
- Check for class imbalance.
- Review incorrect predictions and low-confidence results.
- Make sure training crops represent the appearance of civets in actual CCTV footage.

**Movement and activity issues**
- Review movement threshold settings if bounding-box jitter is incorrectly interpreted as locomotion.
- Check transition confirmation and unknown tolerances if activity states change too frequently or take too long to update.
- Evaluate the system on longer footage to identify problems that may not appear in short clips.

Avoid judging the system solely by confidence scores or validation metrics. Strong validation results do not guarantee reliable performance on unfamiliar CCTV footage.

## Notes

- Frame extraction, enhancement, annotation, training, and inference are separate stages.
- Keep original CCTV footage unchanged.
- Apply consistent preprocessing to training and inference data.
- Split datasets by source video to prevent data leakage.
- Keep the test set separate from training and validation data.
- The behavior pipeline currently assumes one civet in the monitored environment.
- Do not commit large CCTV videos, datasets, or trained model weights to Git unless necessary and permitted.