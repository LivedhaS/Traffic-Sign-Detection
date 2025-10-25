import argparse
import time
from pathlib import Path
import cv2
import torch
import pyttsx3
import datetime
import threading
import queue
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, set_logging  # Add set_logging import
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_one_box

# Initialize pyttsx3 for TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

frame_queue = queue.Queue(maxsize=10)  # Limit the queue size to avoid memory issues

def display_frame():
    """Function to continuously display frames from the queue."""
    while True:
        if not frame_queue.empty():
            im0 = frame_queue.get()
            # Display the image
            if view_img:
                cv2.imshow('Real-Time Detection', im0)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
    cv2.destroyAllWindows()

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

    save_dir = Path(increment_path(Path("../Results") / opt.name, exist_ok=opt.exist_ok))  # make dir
    set_logging()  # Initialize logging
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # Check image size
    if half:
        model.half()  # Convert model to FP16

    # Load dataset
    dataset = LoadStreams(source, img_size=imgsz) if webcam else LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    last_spoken_class = None

    log_file = open("traffic_sign_log.txt", "a")  # Append mode to keep previous logs

    detection_count = {name: 0 for name in names}

    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert to FP16/FP32
        img /= 255.0  # Normalize to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        for i, det in enumerate(pred):
            im0 = im0s[i].copy() if webcam else im0s

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                    detected_class = names[int(cls)]
                    detection_count[detected_class] += 1

                    if detected_class != last_spoken_class:
                        speak(detected_class)
                        last_spoken_class = detected_class

                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_file.write(f"{timestamp} - Detected: {detected_class}\n")
                        log_file.flush()  # Ensure the log is written immediately

            fps = 1 / (t2 - t1 + 1e-6)  # Avoid division by zero
            cv2.putText(im0, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the image if `view_img` is True
            if view_img:
                cv2.imshow('Real-Time Detection', im0)
                cv2.waitKey(1)  # This ensures the image stays open until manually closed

    print(f'Done. ({time.time() - t0:.3f}s)')

    log_file.write("\nDetection Summary:\n")
    for sign, count in detection_count.items():
        log_file.write(f"{sign}: {count}\n")
    log_file.write("\n")
    log_file.close()

    if save_img:
        if dataset.mode == 'images':
            cv2.imwrite(str(save_dir / Path(path).name), im0)
        else:  # 'video' mode
            if vid_cap:
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(str(save_dir / Path(path).name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../Model/Model/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-img', action='store_true', help='save results to images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

    if opt.save_txt or opt.save_img:
        print('Results saved to %s' % save_dir)
