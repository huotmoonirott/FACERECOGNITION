"""
camera_speed_test.py — double-click this file (or run it) and click the
button. No command line typing needed.

It tests your camera (index 1, the one your main app actually uses) at
several resolutions and tells you the real measured FPS for each — then
tells you in plain English whether 144fps is realistic on this hardware.
"""

import time
import threading
import tkinter as tk
from tkinter import scrolledtext
# pyrefly: ignore [missing-import]
import cv2

CAM_INDEX = 1  # your main app already found the real camera at index 1

MODES_TO_TEST = [
    (1920, 1080, 30),
    (1280, 720,  60),
    (1280, 720,  30),
    (640,  480,  120),
    (640,  480,  60),
    (640,  480,  30),
    (320,  240,  120),
    (320,  240,  60),
]

TEST_SECONDS = 2.0


def probe_mode(w, h, fps_request):
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps_request)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(5):
        cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    start = time.time()
    while time.time() - start < TEST_SECONDS:
        ret, _ = cap.read()
        if ret:
            count += 1
    elapsed = time.time() - start
    measured_fps = count / elapsed if elapsed > 0 else 0.0
    cap.release()
    return (w, h, fps_request, actual_w, actual_h, measured_fps)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Speed Test")
        self.geometry("620x480")
        self.configure(bg="#0C1020")

        tk.Label(
            self, text="Camera Speed Test", bg="#0C1020", fg="#E8EEFF",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(16, 4))

        tk.Label(
            self,
            text="Click the button below. It takes about 16 seconds.\n"
                 "Do not touch the camera or move in front of it while it runs.",
            bg="#0C1020", fg="#6B7A9E", font=("Segoe UI", 10), justify="center"
        ).pack(pady=(0, 12))

        self.btn = tk.Button(
            self, text="▶  Test Camera Speed", font=("Segoe UI", 12, "bold"),
            bg="#3B82F6", fg="white", activebackground="#1D4ED8",
            relief="flat", padx=20, pady=10, command=self.run_test
        )
        self.btn.pack(pady=(0, 12))

        self.output = scrolledtext.ScrolledText(
            self, width=70, height=18, bg="#0F1628", fg="#E8EEFF",
            insertbackground="#E8EEFF", font=("Consolas", 10), relief="flat"
        )
        self.output.pack(padx=16, pady=(0, 16), fill="both", expand=True)
        self.output.insert("end", "Results will appear here.\n")
        self.output.configure(state="disabled")

    def log(self, text):
        self.output.configure(state="normal")
        self.output.insert("end", text + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    def run_test(self):
        self.btn.configure(state="disabled", text="Testing... please wait")
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")
        threading.Thread(target=self._test_thread, daemon=True).start()

    def _test_thread(self):
        self.log(f"Testing camera index {CAM_INDEX} ...\n")
        self.log(f"{'Requested':<16}{'Actual res':<12}{'Measured FPS':<14}")
        self.log("-" * 42)

        results = []
        for w, h, fps in MODES_TO_TEST:
            r = probe_mode(w, h, fps)
            if r is None:
                self.log("Could not open the camera. Is it being used by "
                          "another app (close the main face recognition app first)?")
                self.btn.configure(state="normal", text="▶  Test Camera Speed")
                return
            rw, rh, mfps = r[3], r[4], r[5]
            results.append(r)
            self.log(f"{w}x{h}@{fps:<10}{rw}x{rh:<8}{mfps:<14.1f}")

        best = max(results, key=lambda r: r[5])
        self.log("")
        self.log(f"BEST RESULT: {best[3]}x{best[4]} -> {best[5]:.1f} fps (real, measured)")
        self.log("")
        if best[5] < 35:
            self.log("Your camera tops out around ~30fps no matter the resolution.")
            self.log("This is a hardware limit, not a software problem — most webcams")
            self.log("are 30fps parts. To hit 144fps you'd need different camera")
            self.log("hardware (look for 'high speed' 60-144fps USB3 webcams).")
        else:
            self.log(f"Good news — your camera CAN exceed 30fps, at "
                      f"{best[3]}x{best[4]}.")
            self.log("Send Claude this result and the app's capture resolution")
            self.log("can be updated to unlock this speed.")

        self.btn.configure(state="normal", text="▶  Test Camera Speed")


if __name__ == "__main__":
    App().mainloop()