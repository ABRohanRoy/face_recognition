# File: app/attendance/manager.py
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Set

class AttendanceManager:
    def __init__(self, data_dir: str):
        """
        Initialize attendance manager.
        
        Args:
            data_dir: Directory to store attendance data
        """
        self.data_dir = data_dir
        self.attendance_records = {}  # {student_name: [timestamp1, timestamp2, ...]}
        self.attendance_session_active = False
        self.session_start_time = None
        self.tracked_faces = set()  # To avoid duplicate entries in short time
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def start_session(self) -> None:
        """Start a new attendance tracking session."""
        self.attendance_session_active = True
        self.session_start_time = time.time()
        self.attendance_records = {}
        self.tracked_faces = set()
        print("Attendance session started")
        
    def stop_session(self) -> Dict:
        """
        Stop the current attendance session.
        
        Returns:
            Dictionary of attendance records
        """
        self.attendance_session_active = False
        print("Attendance session stopped")
        return self.get_attendance_summary()
        
    def is_session_active(self) -> bool:
        """Check if attendance session is active."""
        return self.attendance_session_active
        
    def mark_attendance(self, name: str) -> None:
        """
        Mark attendance for a student.
        
        Args:
            name: Name of the student
        """
        if not self.attendance_session_active or name == "Unknown":
            return
            
        # Use a tuple of (name, current_minute) to avoid duplicate entries within a minute
        current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
        tracking_key = (name, current_minute)
        
        if tracking_key in self.tracked_faces:
            return  # Skip if already marked in this minute
            
        # Mark attendance
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if name not in self.attendance_records:
            self.attendance_records[name] = []
            
        self.attendance_records[name].append(timestamp)
        self.tracked_faces.add(tracking_key)
        
        print(f"Marked attendance for {name} at {timestamp}")
        
    def get_attendance_summary(self) -> Dict:
        """
        Get summary of attendance records.
        
        Returns:
            Dictionary with attendance data
        """
        summary = {
            "session_start": datetime.fromtimestamp(self.session_start_time).strftime("%Y-%m-%d %H:%M:%S")
                if self.session_start_time else None,
            "session_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.session_start_time else None,
            "total_students": len(self.attendance_records),
            "records": {name: len(timestamps) for name, timestamps in self.attendance_records.items()},
            "detailed_records": self.attendance_records
        }
        
        return summary
        
    def get_raw_records(self) -> Dict:
        """Get raw attendance records."""
        return self.attendance_records
