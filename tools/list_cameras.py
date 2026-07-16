"""
list_cameras.py — quick check of which camera indices actually open
via DSHOW on this machine. Run this if camera_fps_probe.py fails to
open index 0.

Usage:
    python list_cameras.py
"""

# pyrefly: ignore [missing-import]
import cv2

print("Scanning indices 0-4 ...\n")
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"Index {idx}: OPENS, frame {w}x{h} — looks like a real camera")
        else:
            print(f"Index {idx}: opens but read() failed — likely a phantom/virtual entry")
        cap.release()
    else:
        print(f"Index {idx}: does not open")

print("\nUse the lowest index marked 'looks like a real camera' with camera_fps_probe.py.")