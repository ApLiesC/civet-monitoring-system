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
│   ├── videos/                           # Original or trimmed CCTV videos
│   ├── raw_frames/                       # Extracted frames
│   ├── selected_frames/                  # Frames selected for annotation
│   ├── enhanced_frames/                  # Enhanced images
│   └── civet_dataset/                    # Exported YOLO detection dataset
│
├── models/
│   ├── civet_detector_yolo11n/           # Trained civet detector
│   └── civet_classifier_yolo11n/         # Trained behavior classifier
│
├── scripts/
│   ├── behavior_pipeline/                # Complete behavior monitoring pipeline
│   ├── classification/                   # Behavior classifier training and testing
│   ├── detection/                        # Civet detector training and testing
│   └── preprocessing/                    # Frame extraction and image enhancement
│
├── runs/                                 # Training and inference outputs
├── outputs/
│   └── behavior_pipeline/                # Behavior pipeline results
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

## 7. Prepare the Behavior Classification Dataset

The behavior classification dataset is created using the enhanced frames and the trained civet detection model.

### 7.1 Detect and Crop Civets

1. Use the enhanced frames from `data/enhanced_frames/` as input.
2. Run the trained civet detection model on the enhanced frames.
3. Use the detected bounding boxes to crop the civet from each image.
4. Review the cropped images to ensure that they contain the civet clearly and exclude as much irrelevant background as possible.

The cropped images are then used to create the behavior classification dataset.

### 7.2 Annotate the Cropped Images

Upload the cropped civet images to Roboflow and annotate each image with its corresponding behavior class.

The current classification dataset uses the following classes:

- `eating` – The civet is eating.
- `resting` – The civet is stationary or resting.

Export the annotated dataset in the format required by the classification training script and place it in the appropriate dataset directory.

**Important:** The classification dataset is based on civet crops generated by the detection model, rather than the original full-frame images. This helps the classifier focus on the civet's behavior instead of the surrounding CCTV scene.

## 8. Train the Behavior Classification Model

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

## 9. Test the Models on CCTV Footage

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

## 10. Behavior Monitoring Pipeline

The behavior monitoring pipeline combines the civet detector, movement analyzer, and behavior classifier to process CCTV footage and generate activity and behavior records.

### 10.1 Pipeline Overview

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
        +---------+---------+
        |         |         |
        v         v         v
      Active   Inactive  Not Visible
        |         |         |
        +---------+---------+
                  |
                  v
            Event Tracking
                  |
          +-------+-------+
          |               |
          v               v
    Activity Records  Eating Records
          |               |
          +-------+-------+
                  |
                  v
             Output Files
```

### 10.2 Activity States

The pipeline identifies three overall activity states:

| State | Description |
|---|---|
| `active` | The civet is moving (locomotion) or eating. |
| `inactive` | The civet is detected, stationary, and classified as resting. |
| `not_visible` | The detector cannot locate the civet for longer than the configured tolerance. |

Activity transitions are confirmed over a configurable duration to reduce rapid state changes caused by inconsistent predictions.

Unknown predictions are handled separately rather than automatically being classified as inactive.

### 10.3 Movement Analysis

The movement analyzer uses changes in the center of the civet's bounding box to estimate movement speed.

- `locomotion` – Movement speed reaches or exceeds the configured threshold.
- `stationary` – Movement speed remains below the threshold.
- `unknown` – There is insufficient movement history to determine the state.

Movement analysis helps distinguish locomotion from stationary behaviors such as eating and resting.

### 10.4 Behavior Classification

When the civet is detected and classified as stationary, its image crop is passed to the behavior classifier.

The current classifier recognizes:

- `eating`
- `resting`

Locomotion is determined through movement analysis rather than the behavior classifier.

Predictions below the configured confidence threshold are treated as uncertain.

### 10.5 Eating Event Tracking

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

### 10.6 Pipeline Outputs

After processing the CCTV footage, the behavior monitoring pipeline generates an annotated video and several CSV files.

| Output | Description |
|---|---|
| `annotated_video.mp4` | CCTV footage with detection boxes, behavior labels, and movement information. |
| `behavior_results.csv` | Frame-level detection, movement, classification, and activity information. |
| `activity_records.csv` | Confirmed activity intervals, including active, inactive, and not visible periods. |
| `eating_records.csv` | Recorded eating episodes with start time, end time, and duration. |
| `eating_frequency.csv` | Number of eating episodes within each configured time interval. |

The annotated video displays labels such as:

- `ACTIVE` – The civet is moving.
- `ACTIVE - Eating` – The civet is eating.
- `INACTIVE - Resting` – The civet is resting.
- `NOT VISIBLE` – The civet cannot be detected.

The CSV files contain detailed timestamps and records for further analysis and report generation.

### 10.7 Excel Monitoring Report

The project includes an Excel report generator that uses the pipeline outputs to create a formatted workbook for reviewing civet activity and eating frequency.

The generated workbook is named:

```text
civet_monitoring_report.xlsx
```

The report contains three worksheets.

#### 1. Activity Summary

Displays the civet's recorded activity intervals in three separate groups.

| Activity State | Information |
|---|---|
| Active | Start and stop times of active periods. |
| Inactive | Start and stop times of inactive periods. |
| Not Visible | Start and stop times when the civet cannot be detected. |

Each group has its own sequence numbering, making it easier to review activity periods independently.

#### 2. Eating Frequency

Displays the frequency of recorded eating events over configured time intervals.

| Column | Description |
|---|---|
| Duration | The time interval being analyzed. |
| Frequency - Eating | Number of eating events recorded during that interval. |

The report uses the eating frequency CSV and can also use individual eating-event records as a fallback when frequency counts are unavailable.

#### 3. Visualization

Provides an overview of the CCTV recording and the civet's activity distribution.

It includes:

- **Recording Information** – Recording date, start time, end time, and total video duration.
- **Activity Summary** – Total duration and percentage of time spent in Active, Inactive, and Not Visible states.
- **Activity Distribution** – A pie chart comparing Active and Inactive durations. Not Visible time is excluded from this chart.

#### Report Output

The Excel workbook is saved in the output directory specified when initializing the report generator.

The expected output structure is:

```text
outputs/
└── behavior_pipeline/
    ├── annotated_video.mp4
    ├── behavior_results.csv
    ├── activity_records.csv
    ├── eating_records.csv
    ├── eating_frequency.csv
    └── civet_monitoring_report.xlsx
```

The report generator uses the activity records, eating-event records, eating-frequency records, and recording metadata to produce the workbook.

Each worksheet is formatted for A4 portrait printing, with the content fitted to one page where possible.

## 11. Configure and Run the Behavior Pipeline

Pipeline settings are stored in:

```text
configs/behavior_pipeline_config.json
```

The configuration includes settings for:

- Video frame sampling.
- Detection confidence and image size.
- Behavior classification confidence.
- Movement threshold and analysis window.
- Activity transition duration.
- Not-visible and unknown-state tolerance.
- Eating event minimum duration and gap tolerance.
- Civet crop padding.
- Eating frequency interval.

Review the configuration before running the pipeline.

Run the pipeline from the project root using:

```bash
python scripts/behavior_pipeline/main.py
```

Use the actual script name and command-line arguments supported by your project if they differ.

## 12. Improve Model Performance

If the system does not perform well on unfamiliar CCTV footage, review the dataset and model outputs before changing training settings.

### Detection Model

- Add more examples from difficult lighting conditions.
- Include images with partial occlusion.
- Review false positives caused by shadows, bushes, and food trays.
- Check bounding box consistency.
- Evaluate on footage from different recording sessions and camera views.

### Classification Model

- Include more examples of eating and resting.
- Make sure each class contains diverse civet positions and orientations.
- Review incorrectly classified crops.
- Check whether the civet is sufficiently visible in the cropped images.
- Include difficult examples from different lighting conditions.

### Behavior Monitoring Pipeline

- Review movement threshold settings.
- Adjust transition duration to reduce rapid state changes.
- Review unknown-state and not-visible tolerances.
- Adjust eating event minimum duration and gap tolerance.
- Compare the generated records with the original CCTV footage.

## Notes

- Keep original CCTV videos unchanged.
- Do not commit large datasets, original CCTV videos, or trained model weights unless required.
- Keep training, validation, and test data separated by source video.
- Use unfamiliar footage to evaluate real-world performance.
- Model paths and output directories may differ depending on configuration and previous runs.