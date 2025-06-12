import cv2
import time
import pygame
import threading
from collections import defaultdict
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # use yolov8m.pt for better accuracy

# Define animal labels and deterrent sounds
animal_sound_map = {
    "cow": "sounds/dog_bark.mp3",
    "sheep": "sounds/wolf_howl.mp3",
    "horse": "sounds/lion_roar.mp3",
    "dog": "sounds/lion_roar.mp3",
    "elephant": "sounds/gun_shot.mp3"  # substitute for lion
}

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Sound playback control
cooldown_time = 15  # total lockout (10s play + 5s rest)
sound_play_duration = 10  # how long to play the sound
last_played = defaultdict(lambda: 0)

# Webcam setup
cap = cv2.VideoCapture(0)

# Function to play sound for a fixed duration in background
def play_sound_continuous(sound_path, duration=10):
    def _play():
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)  # loop
        time.sleep(duration)
        pygame.mixer.music.stop()
    threading.Thread(target=_play, daemon=True).start()

print("Animal deterrent system running. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 prediction
        results = model.predict(source=frame, conf=0.5, verbose=False)[0]

        # Loop through detected objects
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            conf = float(box.conf[0])

            if label in animal_sound_map:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Sound playback logic with cooldown
                current_time = time.time()
                if current_time - last_played[label] > cooldown_time:
                    print(f"Detected {label}. Playing sound for {sound_play_duration} seconds...")
                    play_sound_continuous(animal_sound_map[label], sound_play_duration)
                    last_played[label] = current_time

        # Show camera feed
        cv2.imshow("Animal Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
