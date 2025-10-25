# Traffic-Sign-Detection
📘 Overview

This project presents a Real-Time Traffic Sign Detection and Audio Feedback System that enhances driver awareness and road safety through intelligent sign recognition and voice-based feedback.
Built using YOLOv8 (You Only Look Once), the system accurately detects and classifies traffic signs from live video feeds or images and delivers real-time text-to-speech (TTS) alerts to drivers.

The system’s goal is to reduce distractions and improve responsiveness to road signs — making it suitable for integration with driver assistance and autonomous vehicle technologies.

🚦 Key Features

Real-Time Detection: Fast and accurate traffic sign recognition using YOLOv8.

Audio Feedback: Real-time spoken alerts via pyttsx3 to notify drivers of detected signs.

Multi-Class Recognition: Detects and classifies 61 types of road signs (e.g., Stop, Speed Limit, Pedestrian Crossing).

Dynamic Environment Handling: Performs well under various lighting, weather, and occlusion conditions.

GPU Parallel Processing: Supports high-speed inference using GPU acceleration.

Scalable: Runs on both high-performance and embedded systems.

🧠 System Architecture

The architecture combines deep learning for image-based detection with text-to-speech feedback:

Input Capture: Video or image feed from camera or file.

Detection Model: YOLOv8 performs object detection and classification.

Result Processing: Detected signs are highlighted with bounding boxes and labels.

Audio Output: pyttsx3 converts detected sign labels into voice alerts.

📊 Dataset Description

Total Images: 2,796

Classes: 61 types of traffic signs

Image Size: 640×640 pixels

Annotation Format: YOLO (.txt) — each file includes class label and bounding box coordinates.

Environments: Images captured from highways, urban, and rural roads under varying conditions.

Sample Classes

Stop

Speed Limit

Pedestrian Crossing

No Entry

Roundabout

School Zone

Cyclist Crossing

⚙️ Methodology
1. Data Preprocessing

Images resized to 640×640.

Normalized pixel values between 0–1.

Data augmentation (flipping, rotation, brightness adjustment) to improve generalization.

2. Model Architecture

Based on YOLOv8 — a single-stage, real-time object detector.

Enhanced backbone network for faster, more accurate detections.

3. Training

Model trained for 100 epochs using the Adam optimizer.

80/20 train-validation split.

Data augmentation applied for improved robustness.

4. Evaluation Metrics

Precision: TP / (TP + FP)

Recall: TP / (TP + FN)

mAP@0.5: 91.75%

mAP@0.5:0.95: 74.08%

🧩 Implementation
Requirements

Install dependencies using:

pip install ultralytics pyttsx3 opencv-python

Running Detection

To run the model on a video:

python detect.py --weights "path/to/best.pt" --source "path/to/video.mp4" --view-img


Or on a webcam:

python detect.py --weights "path/to/best.pt" --source 0 --view-img


The system will:

Detect and classify traffic signs.

Draw bounding boxes and labels.

Announce detected signs via TTS.

🔊 Audio Feedback Integration

The audio module uses pyttsx3 for offline text-to-speech conversion.
When a sign is detected (e.g., Speed Limit), the system immediately announces it to the driver.

Example:

import pyttsx3
engine = pyttsx3.init()
engine.say("Speed limit ahead")
engine.runAndWait()

🧾 Results Summary

mAP@0.5: 91.75%

mAP@0.5:0.95: 74.08%

High precision and recall for small and overlapping signs.

Inference Speed: Real-time detection suitable for driver assistance systems.

🧭 Future Scope

Integrate GPS-based sign mapping for context-aware alerts.

Extend to traffic light and lane detection.

Deploy on embedded platforms (e.g., Raspberry Pi with camera).

Support multilingual audio feedback.

🛡️ Conclusion

The proposed Real-Time Traffic Sign Detection and Audio Feedback System successfully combines YOLOv8’s powerful visual recognition with speech-based alerts to create a non-distracting driver assistance tool.
Its robust detection, real-time inference, and adaptability make it a valuable contribution to the development of intelligent transportation and autonomous vehicle systems.
