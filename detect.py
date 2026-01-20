"""
Weapon Detection System
Author: Ali
Date: 2026-01-20
Description: Real-time weapon detection using YOLOv7 and Flask.
"""

import argparse
import time
import sys
import signal
import threading
import os
import webbrowser
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "lib"))  # Add lib to path


import cv2
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from numpy import random
import paho.mqtt.client as mqtt

# Flask
from flask import Flask, render_template, Response, request

# YOLO7 Utils
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    set_logging,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class WeaponDetector:
    def __init__(self):
        self.output_frame = None
        self.lock = threading.Lock()
        self.is_running = True
        self.mqtt_client = self._init_mqtt()
        self.opt = self._get_options()
        self.device = select_device(self.opt.device)
        self.model = self._load_model()

    def _get_options(self):
        opt = pd.Series()
        opt.weights = "best.pt"
        opt.source = "0"
        opt.img_size = 608
        opt.conf_thres = 0.45
        opt.iou_thres = 0.45
        opt.device = "cpu"
        opt.view_img = False  # Headless mode usually
        opt.save_txt = False
        opt.save_conf = False
        opt.nosave = True
        opt.classes = None
        opt.agnostic_nms = True
        opt.augment = True
        opt.project = "runs/detect"
        opt.name = "exp"
        opt.exist_ok = True
        opt.no_trace = True
        return opt

    def _init_mqtt(self):
        try:
            broker_address = "broker.hivemq.com"
            client = mqtt.Client("WEAPON")
            client.connect(broker_address)
            client.loop_start()
            return client
        except Exception as e:
            print(f"Warning: MQTT Connection failed: {e}")
            return None

    def _load_model(self):
        set_logging()
        model = attempt_load(self.opt.weights, map_location=self.device)
        return model

    def stop(self):
        print("Stopping Detector...")
        self.is_running = False
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

    def run(self):
        """Main detection loop (runs in thread)"""
        print(f"Starting detection on source: {self.opt.source}")

        # Load dataset
        stride = int(self.model.stride.max())
        imgsz = check_img_size(self.opt.img_size, s=stride)

        if self.opt.source == "0":
            cudnn.benchmark = True
            dataset = LoadStreams(self.opt.source, img_size=imgsz, stride=stride)
            webcam = True
        else:
            dataset = LoadImages(self.opt.source, img_size=imgsz, stride=stride)
            webcam = False

        # Names and colors
        names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, imgsz, imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )

        for path, img, im0s, vid_cap in dataset:
            if not self.is_running:
                break

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.device.type != "cpu" else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = self.model(img, augment=self.opt.augment)[0]

            # NMS
            pred = non_max_suppression(
                pred,
                self.opt.conf_thres,
                self.opt.iou_thres,
                classes=self.opt.classes,
                agnostic=self.opt.agnostic_nms,
            )

            # Process detections
            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0, frame = path[i], "", im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                if len(det):
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        # MQTT Publish
                        if self.mqtt_client:
                            label = names[int(c)]
                            if label in ["knife", "pistol"]:
                                self.mqtt_client.publish("WEAPON-PUB", str(label))

                    for *xyxy, conf, cls in reversed(det):
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=2,
                        )

                # Update output frame
                with self.lock:
                    self.output_frame = im0.copy()

    def generate_frames(self):
        """Generator for Flask video feed"""
        while self.is_running:
            with self.lock:
                if self.output_frame is None:
                    time.sleep(0.01)
                    continue
                (flag, encodedImage) = cv2.imencode(".jpg", self.output_frame)
                if not flag:
                    continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )


# Initialize Application
app = Flask(__name__)
detector = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    if detector:
        return Response(
            detector.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    return "Detector not ready", 503


@app.route("/shutdown", methods=["POST"])
def shutdown():
    print("Shutdown requested via Web UI")
    cleanup_and_exit()
    return "Server shutting down..."


def cleanup_and_exit(signum=None, frame=None):
    print("\nSignal received! Cleaning up...")
    if detector:
        detector.stop()

    # Needs to run in a separate thread to allow the request to return response before killing
    def kill():
        time.sleep(1)
        os._exit(0)  # Force exit

    threading.Thread(target=kill).start()


def main():
    global detector
    print("Initializing Weapon Detector...")
    detector = WeaponDetector()

    # Start detection thread
    t = threading.Thread(target=detector.run)
    t.daemon = True
    t.start()

    # Auto-open browser
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()

    # Register Signals
    signal.signal(signal.SIGINT, cleanup_and_exit)  # Ctrl+C
    signal.signal(signal.SIGTSTP, cleanup_and_exit)  # Ctrl+Z (converted to exit)

    print("Starting Web Server...")
    app.run(host="0.0.0.0", port="8000", threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
