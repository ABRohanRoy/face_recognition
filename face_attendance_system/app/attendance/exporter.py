# File: app/attendance/exporter.py
import os
import pandas as pd
from datetime import datetime
from typing import Dict

class AttendanceExporter:
    def __init__(self, export_dir: str):
        """
        Initialize exporter.
        
        Args:
            export_dir: Directory to save exported files
        """
        self.export_dir = export_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)
        
    def export_to_excel(self, attendance_data: Dict) -> str:
        """
        Export attendance data to Excel.
        
        Args:
            attendance_data: Dictionary of attendance records
            
        Returns:
            Path to saved Excel file
        """
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for Excel
        records = []
        
        for name, timestamps in attendance_data.items():
            for timestamp in timestamps:
                records.append({
                    "Name": name,
                    "Timestamp": timestamp,
                    "Date": timestamp.split()[0],
                    "Time": timestamp.split()[1]
                })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Save to Excel
        file_path = os.path.join(self.export_dir, f"attendance_{timestamp}.xlsx")
        
        # Create a writer
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Detailed records sheet
            df.to_excel(writer, sheet_name='Detailed Records', index=False)
            
            # Summary sheet
            summary_data = []
            for name, timestamps in attendance_data.items():
                summary_data.append({
                    "Name": name,
                    "First Seen": min(timestamps) if timestamps else None,
                    "Last Seen": max(timestamps) if timestamps else None,
                    "Total Appearances": len(timestamps)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return file_path
