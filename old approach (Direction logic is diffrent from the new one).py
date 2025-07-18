import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import pandas as pd  # <-- Add this import

# --- FOUR REFERENCE POINTS FOR HOMOGRAPHY CALIBRATION ---
# Example pixel coordinates (x, y) in the image (choose corners or known landmarks)
image_points = np.array([
    [100, 100],    # Top-left
    [1820, 100],   # Top-right
    [1820, 980],   # Bottom-right
    [100, 980]     # Bottom-left
], dtype=np.float32)

# Example GPS coordinates (lat, lon) for each point (replace with real values for your use case)
gps_points = [
    (12.971598, 77.594566),   # Top-left
    (12.971598, 77.595566),   # Top-right
    (12.970598, 77.595566),   # Bottom-right
    (12.970598, 77.594566)    # Bottom-left
]

# Convert GPS to local meters using a simple equirectangular projection
# Use the top-left as the reference origin
REF_LAT, REF_LON = gps_points[0]
def gps_to_local_xy(lat, lon, lat0, lon0):
    """Convert GPS (lat, lon) to local (x, y) in meters relative to (lat0, lon0)"""
    delta_lat = (lat - lat0) * 111320
    delta_lon = (lon - lon0) * 111320 * math.cos(math.radians(lat0))
    return delta_lon, delta_lat

# Compute world points in meters for each GPS point
world_points = np.array([
    gps_to_local_xy(lat, lon, REF_LAT, REF_LON) for (lat, lon) in gps_points
], dtype=np.float32)

# Compute homography from image points to world points
H, _ = cv2.findHomography(image_points, world_points)

def pixel_to_world(pt, H):
    """Convert a 2D pixel point to real-world coordinates using homography"""
    px = np.array([pt[0], pt[1], 1.0])
    world_pt = H @ px
    world_pt /= world_pt[2]
    return world_pt[0], world_pt[1]

# --- CONFIGURATION ---
VIDEO_PATH = r"E:\RGB-cameras\3727445-hd_1920_1080_30fps.mp4"  # Path to input video
MODEL_NAME = 'yolov8n.pt'  # You can use yolov8n.pt, yolov8s.pt, etc.
METERS_PER_PIXEL = 0.05  # Calibration factor (meters per pixel)
FPS = 30  # Frames per second of the video
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
SKIP_FRAMES = 2  # Process every 2nd frame for speed

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

    # --- GPS REFERENCE POINT ---
    # Set this to the GPS coordinates corresponding to the (0,0) of your world plane
REF_LAT = 12.971598  # Example: Bangalore
REF_LON = 77.594566

def xy_to_latlon(x, y, lat0, lon0):
    """Convert local (x, y) meters to latitude and longitude."""
    delta_lat = y / 111320
    delta_lon = x / (111320 * math.cos(math.radians(lat0)))
    lat = lat0 + delta_lat
    lon = lon0 + delta_lon
    return lat, lon


# --- MAIN SCRIPT ---
def main():
    # Load YOLOv8 model
    model = YOLO(MODEL_NAME)
    # Enable GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            model.to('cuda')
            print("Using GPU acceleration.")
        else:
            print("GPU not available, using CPU.")
    except ImportError:
        print("Torch not installed, using CPU.")

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

    object_data = defaultdict(lambda: {'lat': [], 'lon': [], 'speed': []})  # <-- Add this line

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            # Frame skipping for optimization
            if frame_idx % SKIP_FRAMES != 0:
                frame_idx += 1
                continue
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

                # Convert centroid to real-world coordinates
                world_x, world_y = pixel_to_world(centroid, H)
                lat, lon = xy_to_latlon(world_x, world_y, REF_LAT, REF_LON)

                # Calculate speed
                speed = calculate_speed(trajectories[object_id], METERS_PER_PIXEL, FPS)
                speed_history[object_id].append(speed)

                # Collect data for Excel output
                object_data[object_id]['lat'].append(lat)
                object_data[object_id]['lon'].append(lon)
                object_data[object_id]['speed'].append(speed)

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

                # Remove object ID from label, only show speed and lat/lon
                label = f"{displayed_speed.get(object_id, 0):.1f} km/h\n({lat}, {lon})"

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

        # Resize frame for smaller GUI window and update GUI every frame
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

    # After the while loop, before cap.release()
    rows = []
    for object_id, data in object_data.items():
        if data['lat'] and data['lon'] and data['speed']:
            avg_lat = sum(data['lat']) / len(data['lat'])
            avg_lon = sum(data['lon']) / len(data['lon'])
            avg_speed = sum(data['speed']) / len(data['speed'])
            rows.append({'id': object_id, 'lat': avg_lat, 'lon': avg_lon, 'speed': avg_speed})

    df = pd.DataFrame(rows, columns=['id', 'lat', 'lon', 'speed'])
    df.to_excel('object_speeds.xlsx', index=False)
    print('Excel file saved as object_speeds.xlsx')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
