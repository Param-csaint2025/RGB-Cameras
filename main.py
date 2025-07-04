import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# --- CONFIGURATION ---
VIDEO_PATH = r"E:\RGB-cameras\3727445-hd_1920_1080_30fps.mp4"  # Path to input video
MODEL_NAME = 'yolov8n.pt'  # You can use yolov8n.pt, yolov8s.pt, etc.
METERS_PER_PIXEL = 0.05  # Calibration factor (meters per pixel)
FPS = 30  # Frames per second of the video
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# --- TRACKER ---
class CentroidTracker:
    def __init__(self, max_lost=30):
        self.next_object_id = 0
        self.objects = dict()  # object_id -> centroid
        self.lost = dict()     # object_id -> lost count
        self.max_lost = max_lost
        self.trajectories = defaultdict(list)  # object_id -> list of (frame_idx, centroid)

    def update(self, detections, frame_idx):
        # detections: list of (x1, y1, x2, y2)
        input_centroids = np.array([
            [(x1 + x2) / 2, (y1 + y2) / 2] for (x1, y1, x2, y2) in detections
        ]) if detections else np.empty((0, 2))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.objects[self.next_object_id] = input_centroids[i]
                self.lost[self.next_object_id] = 0
                self.trajectories[self.next_object_id].append((frame_idx, input_centroids[i]))
                self.next_object_id += 1
            return self.objects.copy()

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        if len(input_centroids) > 0 and len(object_centroids) > 0:
            distances = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            assigned_rows = set()
            assigned_cols = set()

            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.lost[object_id] = 0
                self.trajectories[object_id].append((frame_idx, input_centroids[col]))
                assigned_rows.add(row)
                assigned_cols.add(col)

            # Unassigned existing objects
            for row in set(range(len(object_centroids))) - assigned_rows:
                object_id = object_ids[row]
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    del self.objects[object_id]
                    del self.lost[object_id]
                    del self.trajectories[object_id]

            # New detections
            for col in set(range(len(input_centroids))) - assigned_cols:
                self.objects[self.next_object_id] = input_centroids[col]
                self.lost[self.next_object_id] = 0
                self.trajectories[self.next_object_id].append((frame_idx, input_centroids[col]))
                self.next_object_id += 1
        else:
            # No detections, increment lost count
            for object_id in list(self.lost.keys()):
                self.lost[object_id] += 1
                if self.lost[object_id] > self.max_lost:
                    del self.objects[object_id]
                    del self.lost[object_id]
                    del self.trajectories[object_id]
        return self.objects.copy()

    def get_trajectories(self):
        return self.trajectories

# --- SPEED CALCULATION ---
def calculate_speed(trajectory, meters_per_pixel, fps, window=5):
    # trajectory: list of (frame_idx, centroid)
    if len(trajectory) < 2:
        return 0.0
    # Use last 'window' frames for speed calculation
    points = trajectory[-window:] if len(trajectory) >= window else trajectory
    (f0, c0), (f1, c1) = points[0], points[-1]
    pixel_dist = np.linalg.norm(c1 - c0)
    meter_dist = pixel_dist * meters_per_pixel
    frame_diff = f1 - f0
    if frame_diff == 0:
        return 0.0
    time_sec = frame_diff / fps
    speed_mps = meter_dist / time_sec
    speed_kmh = speed_mps * 3.6
    return speed_kmh

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
    speed_history = defaultdict(lambda: deque(maxlen=10))  # object_id -> last 10 speeds
    displayed_speed = {}  # object_id -> last displayed speed
    last_speed_update = {}  # object_id -> last frame index when speed was updated
    SPEED_UPDATE_INTERVAL = 5  # update speed label every 5 frames

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
                
                # Calculate speed
                speed = calculate_speed(trajectories[object_id], METERS_PER_PIXEL, FPS)
                speed_history[object_id].append(speed)
                
                # Only update displayed speed every SPEED_UPDATE_INTERVAL frames
                if (object_id not in last_speed_update) or (frame_idx - last_speed_update[object_id] >= SPEED_UPDATE_INTERVAL):
                    # Use moving average of last 10 speeds
                    avg_speed = sum(speed_history[object_id]) / len(speed_history[object_id])
                    prev_speed = displayed_speed.get(object_id, avg_speed)
                    # Clamp the speed update to a maximum change of 15 km/h per update
                    MAX_SPEED_CHANGE = 15.0
                    if abs(avg_speed - prev_speed) > MAX_SPEED_CHANGE:
                        if avg_speed > prev_speed:
                            avg_speed = prev_speed + MAX_SPEED_CHANGE
                        else:
                            avg_speed = prev_speed - MAX_SPEED_CHANGE
                    displayed_speed[object_id] = avg_speed
                    last_speed_update[object_id] = frame_idx
                
                label = f"ID {object_id} {displayed_speed.get(object_id, 0):.1f} km/h"
                
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
