"""
Object Tracking, Speed, Direction (Compass), and Lat/Lon Extraction from RGB + Depth Camera Feeds

This proof-of-concept (POC) script demonstrates how to use RGB and depth camera feeds to:
    - Detect and track all visible objects (e.g., vehicles) in a video
    - Estimate each object's real-world speed (km/h)
    - Determine the direction of movement relative to compass (N, NE, E, etc.)
    - Compute the latitude and longitude of each object

Key Features:
    - Uses YOLOv8 for object detection
    - Uses a centroid-based tracker with 3D (depth) support
    - Converts pixel + depth to real-world coordinates using camera intrinsics and extrinsics
    - Computes compass direction using camera facing and object movement
    - Outputs annotated video and saves summary (ID, lat, lon, speed, direction) to Excel

Inputs:
    - RGB video file (VIDEO_PATH)
    - Depth video file (DEPTH_PATH), or None for constant depth (POC/testing only)
    - Camera configuration: latitude, longitude, altitude, facing direction (degrees), intrinsics

DEPTH_PATH details:
    - If set to None, the code uses a constant depth for all detections (for testing pipeline logic)
    - If set to a file path, expects a video (or image sequence) where each frame is a depth map
        - Depth map should be single-channel (grayscale), same size and frame rate as RGB video
        - Pixel values should be proportional to depth (meters or millimeters)
        - Example: 8-bit grayscale, 0 = 0m, 255 = 50m (scaling can be adjusted in code)
    - For other formats (16-bit PNG, proprietary, etc.), adapt the depth loading section as needed

Outputs:
    - Annotated video display (with object ID, class, speed, direction)
    - Excel file ('object_tracking_framewise.xlsx') with frame-wise data for each tracked object

Usage:
    - Update camera parameters and file paths as needed
    - Run the script: python new.py
    - Press ESC to exit the video window

Extensibility:
    - Ready for alerting use-cases (wrong direction, speeding, restricted vehicles, etc.)
    - Modular for integration with live camera feeds or other detectors

Author: [Your Name/Team]
Date: [Date]
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import pandas as pd

# --- CAMERA CONFIGURATION (to be provided for each installation) ---
CAMERA_LAT = 40.712776  # Example: New York City
CAMERA_LON = -74.005974
CAMERA_ALT = 10.0  # meters above sea level (optional, for 3D)
CAMERA_FACING_DEG = 90.0  # 0 = North, 90 = East, 180 = South, 270 = West
CAMERA_FOV_DEG = 90.0  # Field of view (horizontal)
CAMERA_INTRINSICS = {
    "fx": 1000,
    "fy": 1000,
    "cx": 960,
    "cy": 540,  # Example values for 1920x1080
}
FPS = 30  # Frames per second

# --- PATHS ---
VIDEO_PATH = (
    r"D:\Anand\RGB_CAM\RGB-Cameras\4K Video of Highway Traffic!.mp4"  # RGB video
)
DEPTH_PATH = None  # Set to path of depth video or None if not available
MODEL_NAME = "yolov8n.pt"

# --- OBJECT CLASSES TO TRACK (COCO IDs) ---
TRACK_CLASSES = [1, 2, 3, 5, 7]  # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
CLASS_NAMES = {1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}

# --- DISPLAY CONTROL FLAGS ---
SHOW_DIRECTION = True
SHOW_SPEED = False
SHOW_LATLON = False
SHOW_ID = True
SHOW_CLASS = False


# --- UTILITY FUNCTIONS ---
def pixel_depth_to_world(x, y, depth, intrinsics):
    """
    Convert a pixel location and its depth value to 3D camera-centric coordinates (X, Y, Z).

    Args:
        x (int): Pixel x-coordinate (column).
        y (int): Pixel y-coordinate (row).
        depth (float): Depth value at (x, y) in meters.
        intrinsics (dict): Camera intrinsics with keys 'fx', 'fy', 'cx', 'cy'.

    Returns:
        np.ndarray: 3D coordinates (X, Y, Z) in camera frame (meters).
    """
    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])


def camera_to_world(cam_xyz, camera_lat, camera_lon, camera_alt, camera_facing_deg):
    """
    Convert camera-centric 3D coordinates to global ENU (East-North-Up) coordinates and then to latitude/longitude.

    Args:
        cam_xyz (np.ndarray): 3D coordinates in camera frame (meters).
        camera_lat (float): Camera latitude (degrees).
        camera_lon (float): Camera longitude (degrees).
        camera_alt (float): Camera altitude (meters).
        camera_facing_deg (float): Camera facing direction (degrees from North).

    Returns:
        tuple: (lat, lon, alt) - latitude, longitude, and altitude of the point.
    """
    # Rotate according to camera facing
    theta = np.deg2rad(camera_facing_deg)
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    enu = R @ cam_xyz
    # Convert ENU (meters) to lat/lon (approximate, equirectangular)
    d_east, d_north, d_up = enu
    d_lat = d_north / 111320
    d_lon = d_east / (111320 * np.cos(np.radians(camera_lat)))
    lat = camera_lat + d_lat
    lon = camera_lon + d_lon
    alt = camera_alt + d_up
    return lat, lon, alt


def get_compass_direction(dx, dy):
    """
    Compute the movement direction in degrees (0-360, where 0 is North, 90 is East).

    Args:
        dx (float): Displacement in East direction (meters).
        dy (float): Displacement in North direction (meters).

    Returns:
        float: Direction in degrees (0-360, 0 is North, increases clockwise).
    """
    angle = np.arctan2(dx, dy)  # dx=east, dy=north
    angle_deg = (np.degrees(angle) + 360) % 360
    return angle_deg


# --- TRACKER ---
class SimpleTracker:
    """
    Centroid-based tracker for associating object detections across frames using 3D (depth) information.

    Attributes:
        next_id (int): Next object ID to assign.
        objects (dict): Active tracked objects (id -> (centroid_xyz, last_frame)).
        lost (dict): Lost count for each object.
        trajectories (defaultdict): Trajectory history for each object.
        class_ids (dict): Class ID for each object.
        max_lost (int): Max frames to keep lost objects before removing.
    """

    def __init__(self, max_lost=30):
        """
        Initialize the tracker.

        Args:
            max_lost (int): Maximum frames to keep lost objects before deletion.
        """
        self.next_id = 0
        self.objects = dict()  # id -> (centroid_xyz, last_frame)
        self.lost = dict()  # id -> lost count
        self.trajectories = defaultdict(list)  # id -> list of (frame_idx, xyz)
        self.class_ids = dict()  # id -> class_id
        self.max_lost = max_lost

    def update(self, detections, frame_idx):
        """
        Update tracker with new detections for the current frame.

        Args:
            detections (list): List of (xyz, class_id) tuples for detected objects.
            frame_idx (int): Current frame index.

        Returns:
            dict: Updated objects (id -> (centroid_xyz, last_frame)).
        """
        # detections: list of (xyz, class_id)
        input_xyzs = (
            np.array([d[0] for d in detections]) if detections else np.empty((0, 3))
        )
        input_classes = [d[1] for d in detections]
        if len(self.objects) == 0:
            for i, xyz in enumerate(input_xyzs):
                self.objects[self.next_id] = (xyz, frame_idx)
                self.trajectories[self.next_id].append((frame_idx, xyz))
                self.class_ids[self.next_id] = input_classes[i]
                self.lost[self.next_id] = 0
                self.next_id += 1
            return self.objects.copy()
        # Match by nearest neighbor (Euclidean in 3D)
        object_ids = list(self.objects.keys())
        object_xyzs = np.array([self.objects[oid][0] for oid in object_ids])
        if len(input_xyzs) > 0 and len(object_xyzs) > 0:
            dists = np.linalg.norm(object_xyzs[:, None] - input_xyzs[None, :], axis=2)
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]
            assigned_rows, assigned_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                oid = object_ids[row]
                self.objects[oid] = (input_xyzs[col], frame_idx)
                self.trajectories[oid].append((frame_idx, input_xyzs[col]))
                self.class_ids[oid] = input_classes[col]
                self.lost[oid] = 0
                assigned_rows.add(row)
                assigned_cols.add(col)
            # Unassigned existing
            for row in set(range(len(object_xyzs))) - assigned_rows:
                oid = object_ids[row]
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    del self.lost[oid]
                    del self.trajectories[oid]
                    del self.class_ids[oid]
            # New detections
            for col in set(range(len(input_xyzs))) - assigned_cols:
                self.objects[self.next_id] = (input_xyzs[col], frame_idx)
                self.trajectories[self.next_id].append((frame_idx, input_xyzs[col]))
                self.class_ids[self.next_id] = input_classes[col]
                self.lost[self.next_id] = 0
                self.next_id += 1
        else:
            # No detections, increment lost
            for oid in list(self.lost.keys()):
                self.lost[oid] += 1
                if self.lost[oid] > self.max_lost:
                    del self.objects[oid]
                    del self.lost[oid]
                    del self.trajectories[oid]
                    del self.class_ids[oid]
        return self.objects.copy()

    def get_trajectories(self):
        """
        Get the trajectory history for all tracked objects.

        Returns:
            defaultdict: Object trajectories (id -> list of (frame_idx, xyz)).
        """
        return self.trajectories

    def get_class_ids(self):
        """
        Get the class IDs for all tracked objects.

        Returns:
            dict: Object class IDs (id -> class_id).
        """
        return self.class_ids


# --- SPEED CALCULATION ---
def calculate_speed(trajectory, fps, window=5):
    """
    Calculate the speed of an object based on its trajectory in 3D space.

    Args:
        trajectory (list): List of (frame_idx, xyz) tuples for the object.
        fps (float): Frames per second of the video.
        window (int): Number of frames to use for speed calculation (default 5).

    Returns:
        float: Speed in km/h.
    """
    if len(trajectory) < 2:
        return 0.0
    points = trajectory[-window:] if len(trajectory) >= window else trajectory
    (f0, xyz0), (f1, xyz1) = points[0], points[-1]
    dist = np.linalg.norm(xyz1 - xyz0)
    frame_diff = f1 - f0
    if frame_diff == 0:
        return 0.0
    time_sec = frame_diff / fps
    speed_mps = dist / time_sec
    speed_kmh = speed_mps * 3.6
    return speed_kmh


# --- MAIN PIPELINE ---
def main():
    """
    Main pipeline for object detection, tracking, speed, direction, and lat/lon extraction.
    Loads video and depth data, runs detection and tracking, computes all parameters,
    displays annotated video, and saves frame-wise data to Excel.
    """
    # Load YOLOv8 model
    model = YOLO(MODEL_NAME)
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return
    # Optionally open depth video (if available)
    depth_cap = None
    if DEPTH_PATH:
        depth_cap = cv2.VideoCapture(DEPTH_PATH)
    tracker = SimpleTracker(max_lost=30)
    frame_idx = 0
    # Store frame-wise data for each object
    frame_wise_data = []  # List to store all frame-wise object data
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Get depth map for this frame
        if depth_cap:
            ret_d, depth_frame = depth_cap.read()
            if not ret_d:
                break
            # Assume depth_frame is single-channel float32 in meters, same size as frame
            depth_map = (
                cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                / 255.0
                * 50
            )  # Example scaling
        else:
            # For POC: use a constant depth (e.g., 30m) for all detections
            depth_map = np.ones(frame.shape[:2], dtype=np.float32) * 30.0
        # Run YOLOv8 detection
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in TRACK_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                depth = float(depth_map[cy, cx])
                xyz = pixel_depth_to_world(cx, cy, depth, CAMERA_INTRINSICS)
                detections.append((xyz, cls))
        # Update tracker
        objects = tracker.update(detections, frame_idx)
        trajectories = tracker.get_trajectories()
        class_ids = tracker.get_class_ids()
        # For each object, compute lat/lon, speed, direction
        for oid, (xyz, last_frame) in objects.items():
            lat, lon, _ = camera_to_world(
                xyz, CAMERA_LAT, CAMERA_LON, CAMERA_ALT, CAMERA_FACING_DEG
            )
            speed = calculate_speed(trajectories[oid], FPS)
            # Direction: use last two points in trajectory
            traj = trajectories[oid]
            if len(traj) >= 2:
                _, xyz_prev = traj[-2]
                dx, dy = xyz[0] - xyz_prev[0], xyz[1] - xyz_prev[1]
                direction = get_compass_direction(dx, dy)
            else:
                direction = 0.0  # Default to 0 if no trajectory available

            # Store frame-wise data
            frame_data = {
                "frame_number": frame_idx,
                "object_id": oid,
                "class_id": class_ids[oid],
                "class_name": CLASS_NAMES.get(class_ids[oid], "UNKNOWN"),
                "latitude": lat,
                "longitude": lon,
                "speed_kmh": speed,
                "direction_degrees": direction,
            }
            frame_wise_data.append(frame_data)
            # Draw on frame
            label_parts = []
            if SHOW_ID:
                label_parts.append(f"ID:{oid}")
            if SHOW_CLASS:
                label_parts.append(CLASS_NAMES.get(class_ids[oid], "UNK"))
            if SHOW_DIRECTION and len(traj) >= 2:
                label_parts.append(f"{direction:.1f}°")
            if SHOW_SPEED:
                label_parts.append(f"{speed:.1f} km/h")
            if SHOW_LATLON:
                label_parts.append(f"({lat:.6f}, {lon:.6f})")
            label = " | ".join(label_parts)
            # Project xyz back to pixel for display
            fx, fy, cx, cy = (
                CAMERA_INTRINSICS["fx"],
                CAMERA_INTRINSICS["fy"],
                CAMERA_INTRINSICS["cx"],
                CAMERA_INTRINSICS["cy"],
            )
            px = int(xyz[0] * fx / xyz[2] + cx)
            py = int(xyz[1] * fy / xyz[2] + cy)
            # Draw black rectangle as background for text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            rect_top_left = (px, py - text_height - baseline - 4)
            rect_bottom_right = (px + text_width + 4, py + baseline + 4)
            cv2.rectangle(
                frame, rect_top_left, rect_bottom_right, (0, 0, 0), thickness=-1
            )
            # Draw text
            cv2.putText(
                frame,
                label,
                (px, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        cv2.imshow("Tracking (Depth POC)", cv2.resize(frame, None, fx=0.7, fy=0.7))
        if cv2.waitKey(1) & 0xFF == 27:
            break
        frame_idx += 1
    # Save frame-wise data to Excel
    if frame_wise_data:
        df = pd.DataFrame(frame_wise_data)
        # Sort by frame number and object ID for better readability
        df = df.sort_values(["frame_number", "object_id"]).reset_index(drop=True)
        df.to_excel("object_tracking_framewise.xlsx", index=False)
        print(
            f"Frame-wise data saved to object_tracking_framewise.xlsx with {len(df)} records"
        )
        print(f"Total frames processed: {frame_idx + 1}")
        print(f'Unique objects tracked: {len(df["object_id"].unique())}')
    else:
        print("No object data to save")
    cap.release()
    if depth_cap:
        depth_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
