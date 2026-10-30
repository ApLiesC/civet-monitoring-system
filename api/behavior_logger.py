"""Turns per-frame detections into behavior events (start/stop) and saves them to CSV.

Usage inside your detection loop:
    logger = BehaviorLogger("data/events.csv")
    for frame, ts in frames:                 # ts = datetime of the frame
        label = model_prediction_to_label(frame)   # "active", "inactive", "feeding" or None
        logger.update(label, ts)
    logger.close(ts)
"""
import csv
import os
from datetime import datetime

BEHAVIORS = {"active", "inactive", "feeding"}
FIELDS = ["date", "behavior", "start", "stop"]  


class BehaviorLogger:
    def __init__(self, path="data/events.csv", min_seconds=3, max_gap_seconds=5):
        self.path = path
        self.min_seconds = min_seconds        
        self.max_gap_seconds = max_gap_seconds  
        self.label = None
        self.start = None
        self.last_ts = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    def update(self, label, ts: datetime):
        if label not in BEHAVIORS:
            label = None
        gap = self.last_ts and (ts - self.last_ts).total_seconds() > self.max_gap_seconds
        if label != self.label or gap:
            self._flush(self.last_ts or ts)
            self.label, self.start = label, ts
        self.last_ts = ts

    def close(self, ts: datetime):
        self._flush(ts)
        self.label = self.start = None

    def _flush(self, end: datetime):
        if self.label and self.start and (end - self.start).total_seconds() >= self.min_seconds:
            midnight = self.start.replace(hour=0, minute=0, second=0, microsecond=0)
            with open(self.path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.start.date().isoformat(), self.label,
                    int((self.start - midnight).total_seconds()),
                    int((end - midnight).total_seconds()),
                ])
