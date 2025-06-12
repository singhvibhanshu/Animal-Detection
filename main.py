import cv2
import torch
import pygame
import time
from collections import defaultdict

# Load YOLOv5 or YOLOv8 model (ultralytics)
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # use "yolov8m.pt" or "yolov8l.pt" for better accuracy

# Define animal labels and the corresponding deterrent sound
animal_sound_map = {
    "cow": "sounds/dog_bark.mp3",
    "sheep": "sounds/wolf_howl.mp3",
    "horse": "sounds/lion_roar.mp3",
    "dog": "sounds/lion_roar.mp3",
    "lion": "sounds/gun_shot.mp3"
}

# Set up pygame mixer for playing sound
pygame.mixer.init()

# Webcam setup
cap = cv2.VideoCapture(0)

# Cooldown to avoid repetitive sound playing
last_played = defaultdict(lambda: 0)
cooldown_time = 10  # seconds

print("Animal deterrent system is running...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 prediction
        results = model.predict(source=frame, conf=0.5, verbose=False)[0]
        detections = results.names
        boxes = results.boxes

        detected_animals = set()

        for box in boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            detected_animals.add(label)

            if label in animal_sound_map:
                current_time = time.time()
                if current_time - last_played[label] > cooldown_time:
                    print(f"Detected {label}, playing deterrent sound...")
                    sound_path = animal_sound_map[label]
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()
                    last_played[label] = current_time

        # Display result
        cv2.imshow("Animal Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
