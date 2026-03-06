# video_processor.py
"""
Main video processing module for cinema seat occupancy detection.
"""

import cv2
import json
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np

from config import *
from seat_utils import (load_yolo_bboxes, group_and_name_seats, 
                       overlap_iop_iob, apply_seat_mask)
from seat_tracker import SeatTracker
from timestamp_extractor import TimestampExtractor
from csv_writer import CSVWriter


class CinemaSeatProcessor:
    """
    Main processor for cinema seat occupancy detection pipeline.
    """
    
    def __init__(self):
        """Initialize the cinema seat processor with all components."""
        print("🎬 Initializing Cinema Seat Occupancy Detection System...")
        
        # Load models
        self.model = YOLO(YOLO_MODEL_PATH)
        self.timestamp_extractor = TimestampExtractor(TIME_REGION, gpu=True)
        
        # Open video and get properties
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot open video")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {self.width}x{self.height} @ {self.fps:.2f} fps, {self.frame_count} frames")
        
        # Load and process seat data
        self.seat_data = load_yolo_bboxes(SEAT_LABEL_PATH, self.width, self.height)
        self.seat_data, self.seat_names = group_and_name_seats(self.seat_data)
        
        # Create supervision zones for each seat
        self.zones = {s["name"]: sv.PolygonZone(polygon=s["poly"]) for s in self.seat_data}
        
        # Initialize tracker and CSV writer
        self.tracker = SeatTracker(self.seat_names, SMOOTH_WINDOW)
        self.csv_writer = CSVWriter(self.seat_names)
        
        # Initialize video writer
        self.writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO_PATH),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height)
        )
        
        # Timeline data for JSON output
        self.timeline = []
        
        print(f"✅ Initialization complete. Tracking {len(self.seat_names)} seats.")
    
    def process_frame(self, frame, frame_idx):
        """
        Process a single frame for seat occupancy.
        
        Args:
            frame: Video frame
            frame_idx: Current frame index
        
        Returns:
            dict: Seat status for this frame
        """
        # Apply seat mask to focus detection on seat areas only
        masked = apply_seat_mask(frame, self.seat_data)
        
        # Run YOLO detection on masked frame
        results = self.model(
            masked,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]
        
        # Extract person detections
        detections = sv.Detections.from_ultralytics(results)
        person_class_id = [k for k, v in self.model.model.names.items() if v == "person"][0]
        detections = detections[detections.class_id == person_class_id]
        
        # Check occupancy for each seat
        frame_result = {}
        for seat_name, zone in self.zones.items():
            occupied = False
            
            # Check each detection for overlap with this seat
            for i in range(len(detections)):
                iop, iob = overlap_iop_iob(
                    detections.xyxy[i],
                    zone.polygon
                )
                if (iop > IOP_THRESHOLD) or (iob > IOB_THRESHOLD):
                    occupied = True
                    break
            
            # Apply temporal smoothing
            status = "Occupied" if occupied else "Empty"
            smoothed_status = self.tracker.smooth_status(seat_name, status)
            frame_result[seat_name] = smoothed_status
        
        return frame_result
    
    def draw_seats(self, frame):
        """
        Draw seat boundaries and labels on frame.
        
        Args:
            frame: Video frame
        
        Returns:
            Frame with seat annotations
        """
        for s in self.seat_data:
            name = s["name"]
            poly = s["poly"].astype(np.int32)
            
            # Color based on occupancy status
            color = (0, 255, 0) if self.tracker.get_current_status(name) == "Empty" else (0, 0, 255)
            
            # Draw seat polygon
            cv2.polylines(frame, [poly], True, color, 1)
            
            # Draw seat label with background
            x1, y1 = s["x1"], s["y1"]
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(
                frame, name, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        
        return frame
    
    def run(self):
        """
        Main processing loop.
        """
        print(f"🎞 Processing {self.frame_count} frames...")
        
        for frame_idx in tqdm(range(self.frame_count)):
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame at specified stride for efficiency
            if frame_idx % FRAME_STRIDE == 0:
                # Detect seat occupancy
                frame_result = self.process_frame(frame, frame_idx)
                
                # Extract timestamp
                timeline_text, date_str, time_str = self.timestamp_extractor.extract_from_frame(
                    frame, self.fps, frame_idx, FRAME_STRIDE
                )
                
                # Create timeline entry
                timeline_entry = {
                    "timeline": timeline_text,
                    "second": frame_idx // int(self.fps),
                    "date": date_str,
                    "time": time_str,
                    "seats": frame_result,
                    "ticket_purchase": "no",
                    "compliance": "no",
                    "alert": 0,
                    "ticket_number": "",
                    "sms_to_worker": "no"
                }
                self.timeline.append(timeline_entry)
                
                # Add to CSV records at 1-second intervals
                if frame_idx % int(self.fps) == 0:
                    self.csv_writer.add_timestamp_records(
                        date_str, time_str, frame_result, interval_sec=60
                    )
                
                # Add to CSV records at 60-second intervals
                if frame_idx % int(self.fps * 60) == 0:
                    self.csv_writer.add_timestamp_records(
                        date_str, time_str, frame_result, interval_sec=60
                    )
            
            # Draw seat annotations on every frame for output video
            frame = self.draw_seats(frame)
            self.writer.write(frame)
        
        # Cleanup
        self.cap.release()
        self.writer.release()
        
        print("✅ Frame processing complete.")
    
    def save_outputs(self):
        """
        Save all output files (JSON, CSVs).
        """
        # Save JSON timeline
        with open(OUTPUT_JSON_PATH, "w") as f:
            json.dump(self.timeline, f, indent=2)
        
        # Save CSV files
        self.csv_writer.save_csvs(OUTPUT_CSV_1S, OUTPUT_CSV_60S, "60s")
        
        print(f"✅ Annotated video → {OUTPUT_VIDEO_PATH}")
        print(f"✅ JSON timeline  → {OUTPUT_JSON_PATH}")
        print(f"✅ CSV (1s)       → {OUTPUT_CSV_1S}")
        print(f"✅ CSV (60s)      → {OUTPUT_CSV_60S}")


def main():
    """
    Main function to run the cinema seat occupancy detection pipeline.
    """
    processor = CinemaSeatProcessor()
    processor.run()
    processor.save_outputs()
    print("🎬 Processing complete!")


if __name__ == "__main__":
    main()