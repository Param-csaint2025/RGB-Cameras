import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math

from config import VIDEO_PATH, MODEL_NAME, METERS_PER_PIXEL, FPS, VEHICLE_CLASSES, SPEED_UPDATE_INTERVAL, MEDIAN_WINDOW
from tracker import CentroidTracker
from speed import calculate_speed, apply_speed_smoothing

# --- MAIN SCRIPT ---
def main():
    # Load YOLOv8 model
    model = YOLO(MODEL_NAME)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return

    tracker = CentroidTracker(max_lost=30)
    frame_idx = 0
    paused = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # For robust speed display
    speed_history = defaultdict(lambda: deque(maxlen=5))  # object_id -> last 5 raw speeds
    ema_history = defaultdict(lambda: deque(maxlen=MEDIAN_WINDOW))  # object_id -> last N EMA speeds
    prev_ema = defaultdict(float)  # object_id -> previous EMA value
    displayed_speed = {}  # object_id -> last displayed speed
    last_speed_update = {}  # object_id -> last frame index when speed was updated

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # If paused, do not read a new frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

        # Run YOLOv8 inference
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))

        # Update tracker
        objects = tracker.update(detections, frame_idx)
        trajectories = tracker.get_trajectories()

        # Draw detections and speed
        for object_id, centroid in objects.items():
            # Find the bounding box for this centroid
            min_dist = float('inf')
            best_box = None
            for (x1, y1, x2, y2) in detections:
                c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                dist = np.linalg.norm(centroid - c)
                if dist < min_dist:
                    min_dist = dist
                    best_box = (x1, y1, x2, y2)
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate and smooth speed
                speed = calculate_speed(trajectories[object_id], METERS_PER_PIXEL, FPS)
                speed_history[object_id].append(speed)
                smoothed_speed = apply_speed_smoothing(object_id, speed, displayed_speed, prev_ema, ema_history)

                # Only update displayed speed every SPEED_UPDATE_INTERVAL frames
                # and only after at least MEDIAN_WINDOW frames for stability
                if ((object_id not in last_speed_update) or (frame_idx - last_speed_update[object_id] >= SPEED_UPDATE_INTERVAL)) and (frame_idx >= MEDIAN_WINDOW):
                    displayed_speed[object_id] = smoothed_speed
                    last_speed_update[object_id] = frame_idx

                label = f"{displayed_speed.get(object_id, 0):.1f} km/h"
                
                # Draw a filled rectangle as background for the label
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_x = x1
                label_y = y1 - 10
                rect_top_left = (label_x, label_y - text_height - baseline)
                rect_bottom_right = (label_x + text_width, label_y + baseline)
                cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 0, 0), thickness=-1)
                cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw centroid
                cv2.circle(frame, tuple(centroid.astype(int)), 4, (255, 0, 0), -1)

        # Resize frame for smaller GUI window
        display_frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
        cv2.imshow('Vehicle Detection & Speed', display_frame)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 32:  # SPACE to play/pause
            paused = not paused
        elif key == 83 or key == ord('d'):  # RIGHT arrow or 'd' to step forward
            if paused and frame_idx < total_frames - 1:
                frame_idx += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif key == 81 or key == ord('a'):  # LEFT arrow or 'a' to step backward
            if paused and frame_idx > 0:
                frame_idx -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
