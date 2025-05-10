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
    """Background thread to process video frames and recognize faces."""
    global camera, thread_running
    
    if not camera:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            thread_running = False
            return
    
    try:
        # Create a window that can be resized by the user
        cv2.namedWindow('Face Recognition Attendance', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition Attendance', 800, 600)
        
        while thread_running:
            # Read frame
            ret, frame = camera.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                continue
                
            # Detect faces
            face_locations, rgb_small_frame = face_detector.detect_faces(frame)
            
            # Recognize faces
            face_results = face_encoder.recognize_faces(face_locations, rgb_small_frame)
            
            # Mark attendance for recognized faces
            for name, _ in face_results:
                if name != "Unknown":
                    attendance_manager.mark_attendance(name)
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Draw rectangles around faces with names
            for name, (top, right, bottom, left) in face_results:
                # Set color based on recognition status (green for known, red for unknown)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw a box around the face
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                # Draw a label with a name below the face
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(display_frame, name, (left + 6, bottom - 6), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Add attendance status to the frame
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"Attendance Active - {current_time}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            
            # Show number of recognized students
            recognized_count = sum(1 for name, _ in face_results if name != "Unknown")
            total_count = len(face_results)
            cv2.putText(display_frame, f"Recognized: {recognized_count}/{total_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            
            # Add instruction to quit
            cv2.putText(display_frame, "Press 'q' to stop attendance", 
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            
            # Display the resulting frame for visualization
            cv2.imshow('Face Recognition Attendance', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Sleep to reduce CPU usage
            time.sleep(0.05)
            
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()

@router.post("/register", summary="Register a new student with face using live camera")
@router.post("/register", summary="Register a new student with face using live camera")
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