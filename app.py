import os
import cv2
import yaml
import zipfile
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from google.colab import files


# Utility Functions

def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a ZIP file to the specified directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"ZIP file extracted to {extract_to}")
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")

def extract_frames_from_video(video_path: str, output_dir: str, frame_interval: int = 30) -> None:
    """Extract frames from a video at specified intervals."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info - FPS: {fps}, Total Frames: {total_frames}")

    count, saved_count = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"Frame {count} saved as {output_path}")

        count += 1

    cap.release()
    print(f"Total {saved_count} frames saved")

def create_data_yaml(classes_path: str, yaml_path: str) -> None:
    """Create a YAML configuration file for YOLO training."""
    if not os.path.exists(classes_path):
        print(f"classes.txt not found at {classes_path}")
        return

    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    data = {
        'path': '/content/data',
        'train': 'train/images',
        'val': 'validation/images',
        'nc': len(classes),
        'names': classes
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"Created YAML config at {yaml_path}")
    print("\nYAML Contents:")
    with open(yaml_path, 'r') as f:
        print(f.read())

# Object Counter Class

class ObjectCounter:
    """Class to track and count objects crossing a line or entering an area."""
    
    def __init__(self):
        self.tracks = defaultdict(lambda: {"positions": [], "counted_line": False, "counted_area": False})
        self.count_line = 0
        self.count_area = 0
        self.next_track_id = 0

    def get_center(self, box: list) -> tuple:
        """Calculate the center point of a bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def assign_track_id(self, detections: list, frame_count: int) -> None:
        """Assign track IDs to detections based on Euclidean distance."""
        current_centers = [self.get_center(det) for det in detections]
        new_tracks = {}

        if not self.tracks:
            for i, center in enumerate(current_centers):
                new_tracks[self.next_track_id] = {"positions": [center], "counted_line": False, "counted_area": False}
                self.next_track_id += 1
        else:
            prev_centers = [(tid, track["positions"][-1]) for tid, track in self.tracks.items()]
            for center in current_centers:
                min_dist, matched_tid = float('inf'), None
                for tid, prev_center in prev_centers:
                    dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                    if dist < min_dist and dist < 100:
                        min_dist, matched_tid = dist, tid
                if matched_tid is not None:
                    new_tracks[matched_tid] = self.tracks[matched_tid]
                    new_tracks[matched_tid]["positions"].append(center)
                else:
                    new_tracks[self.next_track_id] = {"positions": [center], "counted_line": False, "counted_area": False}
                    self.next_track_id += 1

        for tid, track in new_tracks.items():
            if len(track["positions"]) > 10:
                track["positions"].pop(0)

        self.tracks = new_tracks
        print(f"Frame {frame_count}: Assigned {len(self.tracks)} tracks")

    def is_crossing_line(self, prev_pos: tuple, curr_pos: tuple, line_y: int) -> bool:
        """Check if an object crosses the counting line (top to bottom)."""
        if prev_pos is None or curr_pos is None:
            return False
        _, prev_y = prev_pos
        _, curr_y = curr_pos
        return prev_y < line_y and curr_y >= line_y

    def is_in_area(self, center: tuple, counting_area: tuple) -> bool:
        """Check if the center point is within the counting area."""
        area_x1, area_y1, area_x2, area_y2 = counting_area
        center_x, center_y = center
        return area_x1 <= center_x <= area_x2 and area_y1 <= center_y <= area_y2

    def update(self, detections: list, counting_line: tuple, counting_area: tuple, frame_count: int) -> tuple:
        """Update tracking and counting for line and area."""
        line_start, line_end = counting_line
        _, line_y = line_start

        self.assign_track_id(detections, frame_count)

        for track_id, track in self.tracks.items():
            curr_pos = track["positions"][-1]
            prev_pos = track["positions"][-2] if len(track["positions"]) > 1 else None

            if prev_pos and not track["counted_line"] and self.is_crossing_line(prev_pos, curr_pos, line_y):
                track["counted_line"] = True
                self.count_line += 1
                print(f"Frame {frame_count}: Line count incremented to {self.count_line} for track {track_id}")

            if not track["counted_area"] and self.is_in_area(curr_pos, counting_area):
                track["counted_area"] = True
                self.count_area += 1
                print(f"Frame {frame_count}: Area count incremented to {self.count_area} for track {track_id}")

        return self.count_line, self.count_area

# Video Processing with Counting

def process_video_with_counting(model_path: str, video_path: str, output_path: str, conf_threshold: float = 0.4) -> None:
    """Process video with YOLO detection, tracking, and counting."""
    print("=" * 50)
    print("Starting Video Processing with Counting...")
    print("=" * 50)

    model = YOLO(model_path)
    counter = ObjectCounter()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    # Configure counting line and area
    line_y = int(height * 2/3)
    counting_line = ((0, line_y), (width, line_y))
    area_width, area_height = width // 3, height // 2
    area_x1, area_y1 = width - area_width - 50, height - area_height - 50
    area_x2, area_y2 = area_x1 + area_width, area_y1 + area_height
    counting_area = (area_x1, area_y1, area_x2, area_y2)

    print(f"Counting Line: Y = {line_y} (2/3 height)")
    print(f"Counting Area: ({area_x1}, {area_y1}) to ({area_x2}, {area_y2})")
    print(f"Area Size: {area_width} x {area_height}")

    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()

        detections = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        print(f"Frame {frame_count}: Detected {len(detections)} objects")

        count_line, count_area = counter.update(detections, counting_line, counting_area, frame_count)

        # Draw counting line
        cv2.line(annotated_frame, counting_line[0], counting_line[1], (0, 255, 0), 3)
        cv2.putText(annotated_frame, "COUNTING LINE", (counting_line[0][0] + 10, counting_line[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw counting area
        cv2.rectangle(annotated_frame, (area_x1, area_y1), (area_x2, area_y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "COUNTING AREA", (area_x1, area_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display counts
        cv2.putText(annotated_frame, f"LINE COUNT: {count_line}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"AREA COUNT: {count_area}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"CURRENT DETECTED: {len(detections)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw trails and IDs
        for track_id, track_data in counter.tracks.items():
            positions = track_data["positions"]
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    cv2.line(annotated_frame, positions[i-1], positions[i], (0, 255, 255), 2)
            if positions:
                cv2.putText(annotated_frame, f"ID: {track_id}", (positions[-1][0], positions[-1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(annotated_frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames... Line count: {count_line}, Area count: {count_area}")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")
    print(f"Final Line Count: {count_line}")
    print(f"Final Area Count: {count_area}")

# Video Statistics

def analyze_video_statistics(video_path: str) -> dict:
    """Analyze and return video statistics."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    stats = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()

    print("Video Statistics:")
    print(f"   Resolution: {stats['width']}x{stats['height']}")
    print(f"   FPS: {stats['fps']}")
    print(f"   Total Frames: {stats['total_frames']}")
    print(f"   Duration: {stats['duration']:.2f} seconds")

    return stats

# Main Function

def main():
    """Main function to orchestrate video processing pipeline."""
    print("YOLO OBJECT DETECTION & COUNTING SYSTEM")
    print("=" * 60)

    # Configuration
    zip_path = "/content/box6.zip"
    extract_dir = "/content/extracted_video"
    output_frames_dir = "/content/frames"
    data_zip_path = "/content/data-data-box-box.zip"
    data_extract_dir = "/content/data-box"
    classes_path = "/content/data-box/classes.txt"
    yaml_path = "/content/data.yaml"
    video_path = "/content/box6.mp4"
    model_path = "/content/runs/detect/train/weights/best.pt"
    output_path = "/content/output_box6_counting.mp4"

    # Step 1: Extract ZIP files
    print("Extracting ZIP files...")
    extract_zip(zip_path, extract_dir)
    extract_zip(data_zip_path, data_extract_dir)

    # Step 2: Extract frames from video
    print("\nExtracting frames from video...")
    video_file = None
    for file in os.listdir(extract_dir):
        if file.endswith(('.mp4', '.avi', '.mov')):
            video_file = os.path.join(extract_dir, file)
            break

    if video_file:
        extract_frames_from_video(video_file, output_frames_dir)
    else:
        print(f"Error: No video file found in {extract_dir}")
        return

    # Step 3: Split dataset
    print("\nSplitting dataset...")
    os.system(f"python /content/train_val_split.py --datapath={data_extract_dir} --train_pct=0.9")

    # Step 4: Create YAML configuration
    print("\nCreating YAML configuration...")
    create_data_yaml(classes_path, yaml_path)

    # Step 5: Train YOLO model
    print("\nTraining YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    model.train(data=yaml_path, epochs=50)

    # Step 6: Analyze video
    print("\nAnalyzing video...")
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    analyze_video_statistics(video_path)

    # Step 7: Process video with counting
    print("\nProcessing video with counting...")
    process_video_with_counting(model_path, video_path, output_path, conf_threshold=0.4)

    # Step 8: Download output
    print("\nDownloading output video...")
    files.download(output_path)

    print("\nProcess completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()