
"""Test CCTV timestamp recognition using a video frame."""

import cv2

from cctv_timestamp_reader import CCTVTimestampReader


VIDEO_PATH = "data/videos/locomotion5.mp4"


def main():
    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    reader = CCTVTimestampReader()

    success, frame = capture.read()
    capture.release()

    if not success:
        print("Could not read video frame.")
        return

    timestamp = reader.read(frame)

    print("Recognized timestamp:", timestamp)

    if timestamp is None:
        print("Timestamp recognition failed.")
    else:
        print("Date:", timestamp.strftime("%Y-%m-%d"))
        print("Time:", timestamp.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
