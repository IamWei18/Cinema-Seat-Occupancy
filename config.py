# config.py
"""
Configuration module for Cinema Seat Occupancy Detection System.
Contains all paths and configuration parameters.
"""

from pathlib import Path

# ============================================================
# File Paths Configuration
# ============================================================

# # Video input path
# VIDEO_PATH = ".\cinema-seat-occupancy\source\10.105.71.241_01_2025111219464587.mp4"

# # Seat label file (YOLO format containing seat bounding boxes)
# SEAT_LABEL_PATH = ".\cinema-seat-occupancy\source\vlcsnap-2025-11-14-01h11m36s342.txt"

# # YOLO model weights for person detection
# YOLO_MODEL_PATH = ".\cinema-seat-occupancy\source\people_model.pt"

# # Output directory for all generated files
# OUTPUT_DIR = Path(".\cinema-seat-occupancy\output")
# OUTPUT_DIR.mkdir(exist_ok=True)


# Always anchor all paths to the folder where config.py resides (project root)
BASE_DIR = Path(__file__).resolve().parent

# --- Inputs ---
VIDEO_PATH = BASE_DIR / "source" / "10.105.71.241_01_2025111219464587.mp4"

# Seat label file (YOLO format containing seat bounding boxes)
SEAT_LABEL_PATH = BASE_DIR / "source" / "vlcsnap-2025-11-14-01h11m36s342.txt"

# YOLO model weights for person detection
YOLO_MODEL_PATH = BASE_DIR / "source" / "people_model.pt"

# --- Outputs ---
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure directory exist

# Video name (extracted from path for dynamic file naming)
videoname = Path(VIDEO_PATH).stem

# Output file paths
OUTPUT_VIDEO_PATH = OUTPUT_DIR / f"{videoname}_annotated.mp4"
OUTPUT_JSON_PATH = OUTPUT_DIR / f"{videoname}_cinema_timeline.json"
OUTPUT_CSV_1S = OUTPUT_DIR / f"{videoname}_timeline_1s.csv"
OUTPUT_CSV_60S = OUTPUT_DIR / f"{videoname}_timeline_60s.csv"

# ============================================================
# Detection Parameters
# ============================================================

# Region of CCTV timestamp overlay (x, y, width, height)
# Coordinates are for 1920x1080 resolution
TIME_REGION = (246, 915, 852, 70)  # x, y, w, h

# YOLO detection confidence threshold
CONFIDENCE_THRESHOLD = 0.1

# YOLO IOU threshold for non-maximum suppression
IOU_THRESHOLD = 0.45

# Frame processing stride (process every N frames for efficiency)
FRAME_STRIDE = 5

# Smoothing window for seat status (number of frames to consider)
SMOOTH_WINDOW = 15

# Seat occupancy detection thresholds
IOP_THRESHOLD = 0.15  # Intersection over Polygon threshold
IOB_THRESHOLD = 0.5   # Intersection over Box threshold

# ============================================================
# Seat Layout Configuration
# ============================================================

# Row order from top to bottom (E is top row, A is bottom row)
ROW_LABELS = ["E", "D", "C", "B", "A"]

# Number of seats per row (must match total seats)
ROW_COUNTS = [9, 8, 8, 8, 8]  # E, D, C, B, A