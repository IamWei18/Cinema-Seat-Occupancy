# csv_writer.py
"""
CSV generation module for seat occupancy data.
"""

import pandas as pd
from seat_utils import seat_sort_key, format_date_for_csv


class CSVWriter:
    """
    Generates and manages CSV records for seat occupancy at different intervals.
    """
    
    def __init__(self, seat_names):
        """
        Initialize CSV writer.
        
        Args:
            seat_names (list): List of all seat names
        """
        self.seat_names = sorted(seat_names, key=seat_sort_key)
        self.records_1s = []      # Records for 1-second intervals
        self.records_interval = [] # Records for custom interval (60s)
    
    def add_timestamp_records(self, date_str, time_str, frame_result, interval_sec=60):
        """
        Add records for current timestamp.
        
        Args:
            date_str (str): Date string
            time_str (str): Time string
            frame_result (dict): Seat status dictionary
            interval_sec (int): Interval in seconds for the second CSV
        """
        formatted_date = format_date_for_csv(date_str)
        
        # Create records for all seats at this timestamp
        timestamp_records = []
        for seat in sorted(frame_result.keys(), key=seat_sort_key):
            # Example business logic: mark row C as having purchased tickets
            ticket_purchase_flag = "yes" if seat.startswith("C") else "no"
            compliance_flag = "yes" if seat.startswith("C") else "no"
            
            # Generate ticket number (e.g., #A00001, #B00002)
            ticket_number = f"#{seat[0]}{int(seat[1:]):05d}"
            
            record = {
                "date": formatted_date,
                "time": time_str,
                "seat": seat,
                "status": frame_result[seat],
                "ticket_purchase": ticket_purchase_flag,
                "compliance": compliance_flag,
                "alert": 0,
                "ticket_number": ticket_number,
                "sms_to_worker": "no"
            }
            timestamp_records.append(record)
        
        # Add to 1-second records
        self.records_1s.extend(timestamp_records)
        
        # Add to interval records (e.g., 60 seconds)
        self.records_interval.extend(timestamp_records)
    
    def save_csvs(self, csv_1s_path, csv_interval_path, interval_label="60s"):
        """
        Save CSV files with proper sorting.
        
        Args:
            csv_1s_path (Path): Path for 1-second CSV
            csv_interval_path (Path): Path for interval CSV
            interval_label (str): Label for interval (e.g., "60s")
        """
        # Save 1-second CSV
        if self.records_1s:
            df1 = pd.DataFrame(self.records_1s)
            # Clean time strings
            df1["time"] = df1["time"].apply(self._clean_time_str)
            # Sort by date, time, and seat
            df1 = self._sort_dataframe(df1)
            df1.to_csv(csv_1s_path, index=False)
        else:
            # Write empty CSV with headers
            cols = ["date", "time", "seat", "status", "ticket_purchase", 
                   "compliance", "alert", "ticket_number", "sms_to_worker"]
            pd.DataFrame(columns=cols).to_csv(csv_1s_path, index=False)
        
        # Save interval CSV
        if self.records_interval:
            df_interval = pd.DataFrame(self.records_interval)
            df_interval["time"] = df_interval["time"].apply(self._clean_time_str)
            df_interval = self._sort_dataframe(df_interval)
            df_interval.to_csv(csv_interval_path, index=False)
        else:
            cols = ["date", "time", "seat", "status", "ticket_purchase", 
                   "compliance", "alert", "ticket_number", "sms_to_worker"]
            pd.DataFrame(columns=cols).to_csv(csv_interval_path, index=False)
    
    def _clean_time_str(self, time_str):
        """
        Clean time string by extracting only the time part.
        
        Args:
            time_str (str): Raw time string
        
        Returns:
            str: Cleaned time string
        """
        if not isinstance(time_str, str):
            return ""
        parts = time_str.split()
        for p in parts:
            if ":" in p:
                return p
        return time_str
    
    def _sort_dataframe(self, df):
        """
        Sort DataFrame by date, time, and seat.
        
        Args:
            df (pd.DataFrame): DataFrame to sort
        
        Returns:
            pd.DataFrame: Sorted DataFrame
        """
        # Add temporary columns for sorting
        df["row"] = df["seat"].str[0]
        df["num"] = df["seat"].str[1:].astype(int)
        df.sort_values(by=["date", "time", "row", "num"], inplace=True)
        df.drop(columns=["row", "num"], inplace=True)
        return df