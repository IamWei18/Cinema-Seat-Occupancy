# seat_utils.py
"""
Seat utilities module for handling seat bounding boxes, naming, and overlap detection.
"""

import numpy as np
import cv2
from shapely.geometry import Polygon

# Import configuration
from config import ROW_LABELS, ROW_COUNTS

def load_yolo_bboxes(txt_path, frame_width, frame_height):
    """
    Load YOLO-format seat bounding boxes from text file.
    
    Args:
        txt_path (str): Path to YOLO format text file
        frame_width (int): Video frame width
        frame_height (int): Video frame height
    
    Returns:
        list: List of seat dictionaries containing polygon and coordinates
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()
    
    seats = []
    for line in lines:
        # YOLO format: class x_center y_center width height (all normalized)
        cls, x_c, y_c, w, h = map(float, line.strip().split())
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_c - w / 2) * frame_width)
        y1 = int((y_c - h / 2) * frame_height)
        x2 = int((x_c + w / 2) * frame_width)
        y2 = int((y_c + h / 2) * frame_height)
        
        # Create polygon for the seat (rectangle)
        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        seats.append({
            "poly": poly,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    
    return seats


def group_and_name_seats(seats):
    """
    Group seats into rows and assign structured names (A1, A2, ..., E9).
    
    Args:
        seats (list): List of seat dictionaries with coordinates
    
    Returns:
        tuple: (seats with names, list of seat names)
    """
    # Sort seats by Y coordinate (top to bottom)
    seats_sorted = sorted(seats, key=lambda s: s["y1"])
    
    rows = []
    idx = 0
    
    # Divide sorted seats into predefined rows
    for count in ROW_COUNTS:
        row = seats_sorted[idx:idx + count]
        # Sort each row left to right by X coordinate
        row = sorted(row, key=lambda s: s["x1"])
        rows.append(row)
        idx += count
    
    seat_names = []
    # Assign names: E1-E9, D1-D8, C1-C8, B1-B8, A1-A8
    for label, row in zip(ROW_LABELS, rows):
        for i, seat in enumerate(row):
            seat_name = f"{label}{i + 1}"
            seat["name"] = seat_name
            seat_names.append(seat_name)
    
    return seats, seat_names


def overlap_iop_iob(box, polygon):
    """
    Calculate Intersection over Polygon (IoP) and Intersection over Box (IoB).
    
    Args:
        box: Detection box coordinates [x1, y1, x2, y2]
        polygon: Seat polygon coordinates
    
    Returns:
        tuple: (iop, iob) ratios
    """
    x1, y1, x2, y2 = map(float, box)
    
    # Create polygons for detection box and seat
    box_poly = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    seat_poly = Polygon(polygon)
    
    # Check if polygons are valid
    if not box_poly.is_valid or not seat_poly.is_valid:
        return 0.0, 0.0
    
    # Calculate intersection area
    inter = box_poly.intersection(seat_poly).area
    
    # Calculate ratios
    iop = inter / (seat_poly.area + 1e-6)  # IoP: intersection/seat area
    iob = inter / (box_poly.area + 1e-6)   # IoB: intersection/box area
    
    return iop, iob


def apply_seat_mask(frame, seats):
    """
    Create a mask that only shows seat regions.
    
    Args:
        frame: Input video frame
        seats: List of seat dictionaries with polygons
    
    Returns:
        numpy.ndarray: Masked frame with only seat regions visible
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Fill seat polygons on mask
    for s in seats:
        cv2.fillPoly(mask, [s["poly"].astype(np.int32)], 255)
    
    # Apply mask to frame
    return cv2.bitwise_and(frame, frame, mask=mask)


def seat_sort_key(name):
    """
    Create sort key for seat names (alphabetical then numerical).
    
    Args:
        name (str): Seat name (e.g., 'A1', 'B12')
    
    Returns:
        tuple: Sort key (row letter, seat number)
    """
    if isinstance(name, str) and len(name) > 1 and name[0].isalpha() and name[1:].isdigit():
        return (name[0], int(name[1:]))
    else:
        # Fallback for invalid seat names
        return ("Z", 999)


def format_date_for_csv(date_str):
    """
    Format date string for CSV output (M/D/YYYY).
    
    Args:
        date_str (str): Input date string in various formats
    
    Returns:
        str: Formatted date string
    """
    if not isinstance(date_str, str) or not date_str:
        return date_str
    
    from datetime import datetime
    
    # Try common date patterns
    patterns = ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y"]
    
    for p in patterns:
        try:
            dt = datetime.strptime(date_str, p)
            # Return M/D/YYYY without zero-padding
            return f"{int(dt.month)}/{int(dt.day)}/{dt.year}"
        except Exception:
            continue
    
    # Fallback: return original string
    return date_str