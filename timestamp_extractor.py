# timestamp_extractor.py
"""
Timestamp extraction module using EasyOCR with smart fallback.
"""

import re
from datetime import datetime, timedelta
import easyocr
import cv2


class TimestampExtractor:
    """
    Extracts and tracks timestamp from video overlay using OCR.
    """
    
    def __init__(self, time_region, gpu=True):
        """
        Initialize timestamp extractor.
        
        Args:
            time_region (tuple): (x, y, w, h) of timestamp region
            gpu (bool): Whether to use GPU for OCR
        """
        self.time_region = time_region
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.last_timeline = []
    
    def extract_from_frame(self, frame, fps=None, frame_idx=None, frame_stride=None):
        """
        Extract timestamp from video frame with smart fallback.
        
        Args:
            frame: Video frame
            fps: Frames per second (for time estimation)
            frame_idx: Current frame index (for time estimation)
            frame_stride: Frame processing stride (for time estimation)
        
        Returns:
            tuple: (timeline_text, date_str, time_str)
        """
        x, y, w, h = self.time_region
        roi = frame[y:y + h, x:x + w]
        
        # Perform OCR on timestamp region
        result = self.reader.readtext(roi)
        text_full = " ".join([r[1] for r in result])
        text_full = text_full.replace(".", ":")  # Normalize time separators
        
        # Extract date, time, and day using regex
        date_match = re.search(r"\d{2}-\d{2}-\d{4}", text_full)
        time_match = re.search(r"\d{1,2}:\d{2}:\d{2}", text_full)
        day_match = re.search(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun)", text_full, re.I)
        
        date_str = date_match.group() if date_match else ""
        time_str = time_match.group() if time_match else ""
        day_str = day_match.group() if day_match else ""
        
        # Smart fallback: use previous values if extraction fails
        if self.last_timeline:
            prev = self.last_timeline[-1]
            prev_date = prev.get("date", "")
            prev_time = prev.get("time", "")
            
            if not date_str:
                date_str = prev_date
            
            # Estimate time based on frame progression if time is missing
            if not time_str and prev_time and fps and frame_idx is not None and frame_stride:
                try:
                    prev_dt = datetime.strptime(prev_time.replace('.', ':'), "%H:%M:%S")
                    interval = (frame_stride / fps)
                    est_time = prev_dt + timedelta(seconds=interval)
                    time_str = est_time.strftime("%H:%M:%S")
                except Exception:
                    time_str = prev_time
        
        if not (date_str or time_str):
            return "Unknown", "", ""
        
        timeline_text = f"{date_str} {day_str} {time_str}".strip()
        
        # Store in history for future fallback
        self.last_timeline.append({
            "date": date_str,
            "time": time_str,
            "timeline": timeline_text
        })
        
        return timeline_text, date_str, time_str