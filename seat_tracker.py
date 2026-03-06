# seat_tracker.py
"""
Seat status tracking module with temporal smoothing.
"""

class SeatTracker:
    """
    Tracks seat occupancy status with temporal smoothing to reduce false detections.
    """
    
    def __init__(self, seat_names, smooth_window=15):
        """
        Initialize seat tracker.
        
        Args:
            seat_names (list): List of all seat names
            smooth_window (int): Number of frames to consider for smoothing
        """
        self.smooth_window = smooth_window
        self.seat_history = {name: [] for name in seat_names}
        self.last_status = {name: "Empty" for name in seat_names}
    
    def smooth_status(self, seat_name, current_status):
        """
        Apply temporal smoothing to seat status.
        
        Args:
            seat_name (str): Name of the seat
            current_status (str): Current detected status ("Occupied" or "Empty")
        
        Returns:
            str: Smoothed status
        """
        # Add current status to history
        self.seat_history[seat_name].append(current_status)
        
        # Keep history within window size
        if len(self.seat_history[seat_name]) > self.smooth_window:
            self.seat_history[seat_name].pop(0)
        
        history = self.seat_history[seat_name]
        
        # Smoothing logic: require 5 consecutive "Empty" to change from "Occupied"
        if "Occupied" in history:
            smoothed = "Empty" if history[-5:].count("Empty") == 5 else "Occupied"
        else:
            smoothed = "Empty"
        
        # Update last status
        self.last_status[seat_name] = smoothed
        return smoothed
    
    def get_current_status(self, seat_name):
        """
        Get current smoothed status for a seat.
        
        Args:
            seat_name (str): Name of the seat
        
        Returns:
            str: Current status
        """
        return self.last_status.get(seat_name, "Empty")