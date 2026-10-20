# civet-monitoring-system

## Data Preparation

This stage prepares CCTV footage for civet detection model training.

### Project Structure

```text
civet-monitoring-system/
├── data/
│   ├── videos/             # Original or manually trimmed CCTV videos
│   ├── raw_frames/         # Extracted frames
│   ├── selected_frames/    # Frames selected for annotation
│   └── enhanced_frames/    # Enhanced images for Roboflow
├── scripts/
│   ├── extract_frames.py
│   └── enhance_frames.py
├── requirements.txt
└── README.md
```

### 1. Install Dependencies

Make sure Python is installed, then open a terminal in the project root and run:

```bash
pip install -r requirements.txt
```

### 2. Prepare CCTV Videos

- Place the videos inside `data/videos/`.
- Organize videos into folders by behavior or camera if needed.
- Manually trim unnecessary sections where the civet is absent, too far away, or cannot be reliably identified.

### 3. Extract Frames

Run the frame extraction script:

```bash
python scripts/extract_frames.py
```

The script extracts one frame every 2.5 seconds from each video in `data/videos/` and saves the images in `data/raw_frames/`.

Each filename includes its source video name and timestamp to prevent naming conflicts.

### 4. Select Frames

Review the extracted images and copy useful frames into `data/selected_frames/`.

- Remove unnecessary duplicates and unusable images.
- Keep varied poses, positions, distances, lighting conditions, and identifiable partial occlusions.
- Preserve the source video name and timestamp in each filename.

### 5. Enhance Selected Frames

Run the enhancement script:

```bash
python scripts/enhance_frames.py
```

The script applies grayscale conversion, mild Gaussian smoothing, CLAHE contrast enhancement, and gamma correction to the selected images.

Enhanced images are saved in `data/enhanced_frames/`. The original images remain unchanged.

### 6. Annotate in Roboflow

- Upload the images from `data/enhanced_frames/`.
- Create one object-detection class named `civet`.
- Draw bounding boxes around visible, identifiable civets.
- Review annotations before exporting the dataset in YOLO format.

### Notes

- Frame extraction and image enhancement are separate steps so that only selected images need to be enhanced.
- Compare original and enhanced images before processing the full dataset to ensure that enhancement improves visibility.
- Keep track of each image's source video so the training, validation, and test sets can be separated by video.
- Do not commit large CCTV videos or generated image datasets to Git unless necessary and permitted.
