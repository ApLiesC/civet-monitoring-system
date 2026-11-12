
"""Read CCTV time (required) and date (optional) using RapidOCR."""

import re
from datetime import datetime, time

import cv2
import numpy as np
from rapidocr import RapidOCR


class CCTVTimestampReader:
    def __init__(self, debug=True):
        self.ocr = RapidOCR()
        self.debug = debug

        # Proportional crop coordinates based on the CCTV frame.
        # The Thai weekday between date and time is excluded.
        self.date_region = (0.020, 0.025, 0.145, 0.100)
        self.time_region = (0.180, 0.025, 0.330, 0.100)

        self.upscale_factor = 4

    def _crop_region(self, frame, region):
        height, width = frame.shape[:2]
        x1_ratio, y1_ratio, x2_ratio, y2_ratio = region

        x1 = int(width * x1_ratio)
        y1 = int(height * y1_ratio)
        x2 = int(width * x2_ratio)
        y2 = int(height * y2_ratio)

        return frame[y1:y2, x1:x2]

    @staticmethod
    def _to_grayscale(crop):
        """Convert a crop to grayscale if necessary."""
        if crop.ndim == 2:
            return crop

        if crop.shape[2] == 4:
            return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)

        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def _prepare_original(self, crop):
        """Prepare the original grayscale crop for fast OCR."""
        gray = self._to_grayscale(crop)

        return cv2.resize(
            gray,
            None,
            fx=self.upscale_factor,
            fy=self.upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    def _preprocess_fallback_variations(self, gray):
        """
        Generate additional OCR-friendly images.

        These are only used when OCR on the original crop fails.
        """
        enlarged = cv2.resize(
            gray,
            None,
            fx=self.upscale_factor,
            fy=self.upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )

        # Improve local contrast.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4),
        )

        enhanced = clahe.apply(enlarged)

        # Otsu thresholding.
        _, otsu = cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Adaptive thresholding.
        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        return [
            ("clahe", enhanced),
            ("otsu", otsu),
            ("otsu_inverted", cv2.bitwise_not(otsu)),
            ("adaptive", adaptive),
            ("adaptive_inverted", cv2.bitwise_not(adaptive)),
        ]

    def _run_ocr(self, image, region_name, variation_name):
        """Run RapidOCR and return combined recognized text."""
        try:
            result = self.ocr(image)

            if not result or not result.txts:
                if self.debug:
                    print(
                        f"[OCR] {region_name} ({variation_name}): "
                        "No text detected."
                    )
                return None

            fragments = list(result.txts)
            text = " ".join(fragments)

            if self.debug:
                print(
                    f"\n[OCR] {region_name} ({variation_name})"
                )
                print(f"  Fragments: {fragments}")
                print(f"  Combined:  {text}")
                print(f"  Scores:    {result.scores}")

            return text

        except Exception as error:
            if self.debug:
                print(
                    f"[OCR] {region_name} ({variation_name}) failed: "
                    f"{error}"
                )

            return None

    @staticmethod
    def _extract_time(text):
        """Extract and validate HH:MM:SS."""
        if not text:
            return None

        # Remove spaces that OCR may insert between characters.
        normalized_text = re.sub(r"\s+", "", text)

        match = re.search(
            r"(\d{2})[:.](\d{2})[:.](\d{2})",
            normalized_text,
        )

        if not match:
            return None

        normalized = ":".join(match.groups())

        try:
            return datetime.strptime(
                normalized,
                "%H:%M:%S",
            ).time()
        except ValueError:
            return None

    @staticmethod
    def _extract_date(text):
        """Extract and validate MM-DD-YYYY."""
        if not text:
            return None

        normalized_text = re.sub(r"\s+", "", text)

        match = re.search(
            r"(\d{2})[-/.](\d{2})[-/.](\d{4})",
            normalized_text,
        )

        if not match:
            return None

        normalized = "-".join(match.groups())

        try:
            return datetime.strptime(
                normalized,
                "%m-%d-%Y",
            ).date()
        except ValueError:
            return None

    def _read_region(self, crop, region_name, extractor):
        """
        Read a timestamp region using a fast path and fallback processing.

        The fallback variations are only processed if the original
        grayscale crop does not produce a valid result.
        """
        if crop is None or crop.size == 0:
            if self.debug:
                print(f"[OCR] {region_name}: Empty crop.")

            return None

        gray = self._to_grayscale(crop)

        # -----------------------------------------
        # 1. Fast path: original grayscale image
        # -----------------------------------------

        original = self._prepare_original(gray)

        text = self._run_ocr(
            original,
            region_name,
            "original",
        )

        if text:
            extracted = extractor(text)

            if extracted is not None:
                if self.debug:
                    print(
                        f"[OK] {region_name} extracted "
                        f"from original: {extracted}"
                    )

                return extracted

        # -----------------------------------------
        # 2. Fallback: enhanced variations
        # -----------------------------------------

        if self.debug:
            print(
                f"[OCR] {region_name}: Original OCR unsuccessful. "
                "Trying enhanced images..."
            )

        variations = self._preprocess_fallback_variations(gray)

        for variation_name, processed in variations:
            text = self._run_ocr(
                processed,
                region_name,
                variation_name,
            )

            if not text:
                continue

            extracted = extractor(text)

            if extracted is not None:
                if self.debug:
                    print(
                        f"[OK] {region_name} extracted "
                        f"from {variation_name}: {extracted}"
                    )

                return extracted

        if self.debug:
            print(
                f"[ERROR] Could not extract a valid "
                f"{region_name.lower()}."
            )

        return None

    def _read_time(self, crop):
        """Read the required CCTV time."""
        return self._read_region(
            crop,
            "TIME",
            self._extract_time,
        )

    def _read_date(self, crop):
        """Read the optional CCTV date."""
        return self._read_region(
            crop,
            "DATE",
            self._extract_date,
        )

    def read(self, frame):
        """
        Read CCTV date and time.

        Returns:
            datetime: If both date and time are recognized.
            time: If only time is recognized.
            None: If time recognition fails.
        """
        if frame is None or frame.size == 0:
            if self.debug:
                print("[ERROR] Input frame is empty.")

            return None

        # -----------------------------------------
        # 1. Read time (required)
        # -----------------------------------------

        time_crop = self._crop_region(
            frame,
            self.time_region,
        )

        extracted_time = self._read_time(time_crop)

        if extracted_time is None:
            if self.debug:
                print(
                    "[ERROR] Timestamp extraction failed: "
                    "time is required."
                )

            return None

        # -----------------------------------------
        # 2. Read date (optional)
        # -----------------------------------------

        date_crop = self._crop_region(
            frame,
            self.date_region,
        )

        extracted_date = None

        if date_crop.size > 0:
            extracted_date = self._read_date(date_crop)

        # -----------------------------------------
        # 3. Build return value
        # -----------------------------------------

        if extracted_date is not None:
            timestamp = datetime.combine(
                extracted_date,
                extracted_time,
            )
        else:
            timestamp = extracted_time

        if self.debug:
            print(f"\n[SUCCESS] CCTV time: {extracted_time}")
            print(f"[SUCCESS] CCTV date: {extracted_date}")
            print(f"[SUCCESS] Returned value: {timestamp}")

        return timestamp
