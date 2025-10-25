🚦 Real-Time Traffic Sign Detection and Audio Feedback System
📘 Overview

This project presents a Real-Time Traffic Sign Detection and Audio Feedback System designed to enhance driver awareness and road safety through intelligent sign recognition and voice-based feedback.

Built using YOLOv8 (You Only Look Once), the system detects and classifies traffic signs from live video feeds or images and provides real-time text-to-speech (TTS) alerts to drivers.

The goal is to reduce distractions and improve responsiveness to road signs — suitable for driver-assistance and autonomous vehicle technologies.

🌟 Key Features

Real-Time Detection: Fast and accurate recognition using YOLOv8.

Audio Feedback: Real-time spoken alerts via pyttsx3.

Multi-Class Recognition: Detects 61 traffic sign types (e.g., Stop, Speed Limit, Pedestrian Crossing).

Dynamic Performance: Works under various lighting, weather, and occlusion conditions.

GPU Acceleration: Supports high-speed inference on GPU.

Scalability: Compatible with both high-end and embedded systems.

🧠 System Architecture

Input Capture: Video or image input from camera or file.

Detection Model: YOLOv8 performs sign detection and classification.

Result Processing: Detected signs are annotated with bounding boxes and labels.

Audio Output: pyttsx3 generates voice alerts for detected signs.

📊 Dataset Description

Total Images: 2,796

Classes: 61 traffic sign types

Image Size: 640×640 pixels

Annotation Format: YOLO (.txt)

Environments: Highway, urban, and rural scenarios under varying conditions

Sample Classes:
Stop | Speed Limit | Pedestrian Crossing | No Entry | Roundabout | School Zone | Cyclist Crossing

⚙️ Methodology
1. Data Preprocessing

Resized all images to 640×640.

Normalized pixel values between 0–1.

Applied augmentation (flipping, rotation, brightness adjustments).

2. Model Architecture

Based on YOLOv8, a single-stage real-time object detector.

Enhanced backbone for improved accuracy and speed.

3. Training

Epochs: 100

Optimizer: Adam

Split: 80% training, 20% validation

Data augmentation for robustness.

4. Evaluation Metrics

Precision: TP / (TP + FP)

Recall: TP / (TP + FN)

mAP@0.5: 91.75%

mAP@0.5:0.95: 74.08%

🧩 Implementation Requirements
Dependencies

Install all dependencies using:

pip install ultralytics pyttsx3 opencv-python

🚀 Running Detection
Run on a Video
python detect.py --weights "path/to/best.pt" --source "path/to/video.mp4" --view-img

Run on a Webcam
python detect.py --weights "path/to/best.pt" --source 0 --view-img

The System Will

Detect and classify traffic signs.

Draw bounding boxes and labels.

Announce detected signs via TTS.

🔊 Audio Feedback Integration

Uses pyttsx3 for offline TTS conversion. When a sign is detected (e.g., Speed Limit), the system immediately announces it.

Example:

import pyttsx3
engine = pyttsx3.init()
engine.say("Speed limit ahead")
engine.runAndWait()

🧾 Results Summary
Metric	Value
mAP@0.5	91.75%
mAP@0.5:0.95	74.08%
Inference Speed	Real-time (suitable for driver assistance)
Performance	High precision and recall even for small or overlapping signs
🧭 Future Scope

GPS-based sign mapping for context-aware alerts.

Integration with traffic light and lane detection systems.

Deployment on embedded devices (e.g., Raspberry Pi).

Support for multilingual voice alerts.

🛡️ Conclusion

The Real-Time Traffic Sign Detection and Audio Feedback System effectively combines YOLOv8’s advanced visual recognition with real-time speech alerts, creating a non-distracting, intelligent driver-assistance tool.

Its robust detection, high-speed inference, and adaptability make it a valuable step toward smart transportation and autonomous vehicle systems.
