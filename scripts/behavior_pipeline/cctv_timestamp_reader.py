
"""Read date and time from the CCTV timestamp overlay using RapidOCR."""

import re
from datetime import datetime

import cv2
from rapidocr import RapidOCR


class CCTVTimestampReader:
    def __init__(self, debug=True):
        self.ocr = RapidOCR()
        self.debug = debug

        # Proportional crop coordinates based on the CCTV frame.
        # The Thai weekday between date and time is excluded.
        self.date_region = (0.020, 0.025, 0.145, 0.100)
        self.time_region = (0.180, 0.025, 0.330, 0.100)

    def _crop_region(self, frame, region):
        height, width = frame.shape[:2]
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = region

        x1 = int(width * x1_ratio)
        y1 = int(height * y1_ratio)
        x2 = int(width * x2_ratio)
        y2 = int(height * y2_ratio)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _preprocess(crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        enlarged = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

        _, threshold = cv2.threshold(
            enlarged,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return cv2.bitwise_not(threshold)

    def _read_text(self, crop, region_name):
        processed = self._preprocess(crop)
        result = self.ocr(processed)

        if not result or not result.txts:
            if self.debug:
                print(f"[OCR] {region_name}: No text detected.")
            return "", []

        fragments = list(result.txts)

        # Join OCR fragments and remove duplicated separators
        # created where adjacent fragments overlap.
        text = "".join(fragments)
        text = re.sub(r"[-/]{2,}", "-", text)

        if self.debug:
            print(f"\n[OCR] {region_name}")
            print(f"  Fragments: {fragments}")
            print(f"  Combined:  {text}")
            print(f"  Scores:    {result.scores}")

        return text, fragments

    def read(self, frame):
        if frame is None or frame.size == 0:
            print("[ERROR] Input frame is empty.")
            return None

        date_crop = self._crop_region(frame, self.date_region)
        time_crop = self._crop_region(frame, self.time_region)

        if date_crop.size == 0 or time_crop.size == 0:
            print("[ERROR] Date or time crop is empty.")
            return None

        date_text, _ = self._read_text(date_crop, "DATE")
        time_text, _ = self._read_text(time_crop, "TIME")

        date_match = re.search(
            r"\d{2}[-/]\d{2}[-/]\d{4}",
            date_text,
        )

        time_match = re.search(
            r"\d{2}:\d{2}:\d{2}",
            time_text,
        )

        if not date_match:
            print(f"\n[ERROR] Could not extract date from: {date_text!r}")
        else:
            print(f"\n[OK] Extracted date: {date_match.group()}")

        if not time_match:
            print(f"[ERROR] Could not extract time from: {time_text!r}")
        else:
            print(f"[OK] Extracted time: {time_match.group()}")

        if not date_match or not time_match:
            print("[ERROR] Timestamp extraction failed.")
            return None

        date_text = date_match.group().replace("/", "-")
        time_text = time_match.group()

        combined_timestamp = f"{date_text} {time_text}"

        if self.debug:
            print(f"\n[PARSE] Attempting to parse: {combined_timestamp}")
            print("[PARSE] Expected format: MM-DD-YYYY HH:MM:SS")

        try:
            timestamp = datetime.strptime(
                combined_timestamp,
                "%m-%d-%Y %H:%M:%S",
            )

            if self.debug:
                print(f"[SUCCESS] Parsed timestamp: {timestamp}")

            return timestamp

        except ValueError as error:
            print(f"[ERROR] Date/time parsing failed: {error}")
            return None
