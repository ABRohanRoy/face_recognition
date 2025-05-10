# File: app/api/endpoints.py
import os
import threading
import time
from typing import Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
import cv2
import shutil
from datetime import datetime

from app.face_recognition.face_detector import FaceDetector
from app.face_recognition.face_encoder import FaceEncoder
from app.attendance.manager import AttendanceManager
from app.attendance.exporter import AttendanceExporter

router = APIRouter()

# Global state
camera = None
recognition_thread = None
thread_running = False

# Initialize components
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDENTS_DIR = os.path.join(BASE_DIR, "data", "students")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "data", "attendance_records")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp")

# Create directories
os.makedirs(STUDENTS_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize components
face_detector = FaceDetector()
face_encoder = FaceEncoder(STUDENTS_DIR)
attendance_manager = AttendanceManager(ATTENDANCE_DIR)
attendance_exporter = AttendanceExporter(ATTENDANCE_DIR)

def process_frames():
    """Background thread to process video frames and recognize faces with Apple-inspired UI."""
    global camera, thread_running
    
    if not camera:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            thread_running = False
            return
    
    try:
        # Create a window that can be properly seen
        cv2.namedWindow('Attendance System', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Attendance System', 1024, 768)
        
        # Define UI colors (Apple-inspired)
        APPLE_BG = (240, 240, 247)     # Light gray background
        APPLE_BLUE = (255, 149, 0)     # Apple Music orange (BGR format)
        APPLE_TEXT = (50, 50, 50)      # Dark gray text
        APPLE_GREEN = (0, 200, 0)      # Green for known faces
        APPLE_RED = (0, 0, 255)        # Red for unknown faces
        APPLE_ACCENT = (255, 200, 0)   # Light orange for accents
        
        font = cv2.FONT_HERSHEY_DUPLEX
        
        while thread_running:
            # Read frame
            ret, frame = camera.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                continue
                
            # Create the main UI frame
            h, w = frame.shape[:2]
            ui_frame = np.full((h + 140, w, 3), APPLE_BG, dtype=np.uint8)
            
            # Add top header bar
            cv2.rectangle(ui_frame, (0, 0), (w, 60), APPLE_BLUE, -1)
            cv2.putText(ui_frame, "Attendance System", (20, 40), 
                        font, 1, (255, 255, 255), 1)
            
            current_time = datetime.now().strftime("%H:%M:%S")
            current_date = datetime.now().strftime("%Y-%m-%d")
            time_width = cv2.getTextSize(current_time, font, 0.8, 1)[0][0]
            cv2.putText(ui_frame, current_time, (w - time_width - 20, 30), 
                        font, 0.8, (255, 255, 255), 1)
            cv2.putText(ui_frame, current_date, (w - time_width - 20, 50), 
                        font, 0.6, (255, 255, 255), 1)
            
            # Place camera feed in the main area
            ui_frame[70:70+h, 0:w] = frame
                
            # Detect faces
            face_locations, rgb_small_frame = face_detector.detect_faces(frame)
            
            # Recognize faces
            face_results = face_encoder.recognize_faces(face_locations, rgb_small_frame)
            
            # Mark attendance for recognized faces
            recognized_names = []
            for name, _ in face_results:
                if name != "Unknown":
                    attendance_manager.mark_attendance(name)
                    recognized_names.append(name)
            
            # Draw rectangles around faces with names (Modern, minimal style)
            for name, (top, right, bottom, left) in face_results:
                # Set color based on recognition status
                color = APPLE_GREEN if name != "Unknown" else APPLE_RED
                
                # Draw a box around the face (in the UI frame)
                cv2.rectangle(ui_frame, (left, top+70), (right, bottom+70), color, 2)
                
                # Add a label with the name (minimal design)
                label_y = bottom + 70 + 35
                cv2.rectangle(ui_frame, (left, bottom+70), (right, label_y), color, cv2.FILLED)
                
                # Ensure text fits within the label
                scale = 0.6
                name_width = cv2.getTextSize(name, font, scale, 1)[0][0]
                while name_width > (right - left - 10) and scale > 0.3:
                    scale -= 0.05
                    name_width = cv2.getTextSize(name, font, scale, 1)[0][0]
                
                text_x = left + (right - left - name_width) // 2
                cv2.putText(ui_frame, name, (text_x, bottom+70+25), 
                            font, scale, (255, 255, 255), 1)
            
            # Add bottom info bar
            cv2.rectangle(ui_frame, (0, h+80), (w, h+140), (255, 255, 255), -1)
            
            # Add statistics and instructions
            recognized_count = sum(1 for name, _ in face_results if name != "Unknown")
            total_count = len(face_results)
            
            cv2.putText(ui_frame, f"Recognized: {recognized_count}/{total_count}", 
                        (20, h+110), font, 0.7, APPLE_TEXT, 1)
            
            # Show recently detected students
            if recognized_names:
                recent_text = "Present: " + ", ".join(recognized_names[:3])
                if len(recognized_names) > 3:
                    recent_text += f" +{len(recognized_names)-3} more"
                
                # Calculate text width to align properly
                text_width = cv2.getTextSize(recent_text, font, 0.6, 1)[0][0]
                if text_width > w - 300:  # Truncate if too long
                    recent_text = recent_text[:40] + "..."
                
                cv2.putText(ui_frame, recent_text, 
                            (20, h+130), font, 0.6, APPLE_TEXT, 1)
            
            # Add instruction to quit
            quit_text = "Press 'q' to stop"
            text_width = cv2.getTextSize(quit_text, font, 0.7, 1)[0][0]
            cv2.putText(ui_frame, quit_text, 
                        (w - text_width - 20, h+110), font, 0.7, APPLE_RED, 1)
            
            # Display the final UI frame
            cv2.imshow('Attendance System', ui_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Sleep to reduce CPU usage
            time.sleep(0.05)
            
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()

async def register_student(name: str = Form(...)):
    """
    Register a new student by capturing their face from the webcam.
    """
    try:
        print(f"Starting registration process for student: {name}")
        
        # Use the FaceEncoder's live registration method
        success = face_encoder.register_face_live(name)
        
        if success:
            return {"message": f"Student {name} registered successfully"}
        else:
            raise HTTPException(status_code=400, detail="Registration failed or was cancelled")
    except Exception as e:
        print(f"Error during registration: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")
    
@router.post("/start-attendance", summary="Start attendance tracking")
async def start_attendance(background_tasks: BackgroundTasks):
    """
    Start attendance tracking using the webcam.
    """
    global recognition_thread, thread_running
    
    if thread_running:
        return {"message": "Attendance tracking is already running"}
    
    # Start attendance session
    attendance_manager.start_session()
    
    # Start recognition in a background thread
    thread_running = True
    recognition_thread = threading.Thread(target=process_frames)
    recognition_thread.daemon = True
    recognition_thread.start()
    
    return {"message": "Attendance tracking started"}

@router.post("/stop-attendance", summary="Stop attendance tracking")
async def stop_attendance():
    """
    Stop attendance tracking and return summary.
    """
    global thread_running, camera
    
    if not thread_running:
        return {"message": "Attendance tracking is not running"}
    
    # Stop the thread
    thread_running = False
    
    # Wait for the thread to finish
    if recognition_thread:
        recognition_thread.join(timeout=2.0)
    
    # Release camera
    if camera:
        camera.release()
        camera = None
    
    # Stop attendance session and get summary
    summary = attendance_manager.stop_session()
    
    return {
        "message": "Attendance tracking stopped",
        "summary": summary
    }

@router.get("/attendance-status", summary="Get current attendance status")
async def get_attendance_status():
    """
    Get the current status of attendance tracking.
    """
    is_active = attendance_manager.is_session_active()
    
    response = {
        "is_active": is_active,
        "message": "Attendance tracking is active" if is_active else "Attendance tracking is not active"
    }
    
    if is_active:
        response["summary"] = attendance_manager.get_attendance_summary()
    
    return response

@router.get("/export-attendance", summary="Export attendance to Excel")
async def export_attendance():
    """
    Export current attendance data to Excel.
    """
    attendance_data = attendance_manager.get_raw_records()
    
    if not attendance_data:
        raise HTTPException(status_code=404, detail="No attendance data available")
    
    excel_path = attendance_exporter.export_to_excel(attendance_data)
    
    return FileResponse(
        path=excel_path, 
        filename=os.path.basename(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@router.get("/students", summary="Get list of registered students")
async def get_students():
    """
    Get a list of all registered students.
    """
    return {"students": face_encoder.known_face_names}