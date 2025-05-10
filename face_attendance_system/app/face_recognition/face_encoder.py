import os
import pickle
import face_recognition
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
import time

class FaceEncoder:
    def __init__(self, encodings_dir: str, tolerance: float = 0.6):
        """
        Initialize face encoder.
        
        Args:
            encodings_dir: Directory to save face encodings
            tolerance: Threshold for face comparison (lower is stricter)
        """
        self.encodings_dir = encodings_dir
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Create directory if it doesn't exist
        os.makedirs(self.encodings_dir, exist_ok=True)
        
        # Load existing encodings
        self.load_encodings()
        
    def load_encodings(self) -> None:
        """Load all saved face encodings from the directory."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.encodings_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.encodings_dir, filename)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings.append(data['encoding'])
                    self.known_face_names.append(data['name'])
                    
        print(f"Loaded {len(self.known_face_names)} face encodings")
        
    def encode_face(self, image_path: str, name: str) -> bool:
        """
        Create encoding for a new face.
        
        Args:
            image_path: Path to the image containing the face
            name: Name of the person
            
        Returns:
            True if successful, False otherwise
        """
        # Load image and convert to RGB
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image at {image_path}")
            return False
            
        # Convert BGR to RGB (important for face_recognition)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            print("No face detected in the image")
            return False
        
        # Use the first face found
        face_encoding = face_recognition.face_encodings(rgb_image, [face_locations[0]])[0]
        
        # Save encoding
        student_id = name.lower().replace(' ', '_')
        encoding_data = {
            'name': name,
            'encoding': face_encoding,
            'created_at': time.time()
        }
        
        # Save to file
        encoding_file = os.path.join(self.encodings_dir, f"{student_id}.pkl")
        with open(encoding_file, 'wb') as f:
            pickle.dump(encoding_data, f)
            
        # Update in-memory encodings
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        
        return True

    def register_face_live(self, name: str) -> bool:
        """
        Register a face using live camera feed with Apple-inspired UI.
        
        Args:
            name: Name of the person
            
        Returns:
            True if successful, False otherwise
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Could not open webcam")
            return False
        
        face_captured = False
        capture_confirmed = False
        temp_dir = os.path.join(os.path.dirname(self.encodings_dir), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_image_path = os.path.join(temp_dir, f"{name}_{int(time.time())}.jpg")
        
        captured_frame = None
        
        # Define UI colors (Apple-inspired)
        APPLE_BG = (240, 240, 247)  # Light gray background
        APPLE_BLUE = (255, 149, 0)  # Apple music orange (BGR format)
        APPLE_TEXT = (50, 50, 50)   # Dark gray text
        APPLE_RED = (0, 0, 255)     # Red for cancel
        APPLE_GREEN = (0, 200, 0)   # Green for confirm
        
        try:
            # Create a window that can be properly seen
            cv2.namedWindow('Face Registration', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Face Registration', 800, 600)
            
            # State management
            preview_mode = False
            
            while True:
                if not preview_mode:
                    # Camera capture mode
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Create a clean UI frame
                    ui_frame = np.full((frame.shape[0], frame.shape[1], 3), APPLE_BG, dtype=np.uint8)
                    
                    # Add a centered camera view
                    h, w = frame.shape[:2]
                    offset_y = 60  # Space for the top bar
                    
                    # Display the camera feed
                    ui_frame[offset_y:offset_y+h, 0:w] = frame
                    
                    # Detect faces in current frame
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    # Draw face detection indicators
                    for top, right, bottom, left in face_locations:
                        # Scale back up face locations
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Draw rounded rectangle around face
                        cv2.rectangle(ui_frame, (left, top+offset_y), (right, bottom+offset_y), APPLE_BLUE, 2)
                    
                    # Top title bar
                    cv2.rectangle(ui_frame, (0, 0), (w, 50), APPLE_BLUE, -1)
                    cv2.putText(ui_frame, f"Register: {name}", (20, 35), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
                    
                    # Bottom control bar
                    cv2.rectangle(ui_frame, (0, h+offset_y), (w, h+offset_y+50), (255, 255, 255), -1)
                    
                    # Action buttons
                    if face_locations:
                        cv2.putText(ui_frame, "Face Detected", (20, h+offset_y+35), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_GREEN, 1)
                        cv2.putText(ui_frame, "Press [C] to Capture", (w-250, h+offset_y+35), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_BLUE, 1)
                    else:
                        cv2.putText(ui_frame, "No Face Detected", (20, h+offset_y+35), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_RED, 1)
                    
                    cv2.putText(ui_frame, "Press [Q] to Cancel", (w//2-100, h+offset_y+35), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_RED, 1)
                    
                    # Show UI frame
                    cv2.imshow('Face Registration', ui_frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    
                    # If 'c' pressed, capture the current frame
                    if key == ord('c'):
                        if face_locations:
                            captured_frame = frame.copy()
                            cv2.imwrite(temp_image_path, captured_frame)
                            face_captured = True
                            preview_mode = True
                            print("Face captured! Press 'y' to confirm or 'r' to retake.")
                        else:
                            print("No face detected. Please position your face in the camera.")
                    
                    # If 'q' pressed, cancel registration
                    elif key == ord('q'):
                        print("Registration cancelled")
                        break
                else:
                    # Preview mode - showing captured image
                    preview_frame = np.full((captured_frame.shape[0]+110, captured_frame.shape[1], 3), 
                                          APPLE_BG, dtype=np.uint8)
                    
                    # Display captured image
                    h, w = captured_frame.shape[:2]
                    offset_y = 60
                    preview_frame[offset_y:offset_y+h, 0:w] = captured_frame
                    
                    # Top title bar
                    cv2.rectangle(preview_frame, (0, 0), (w, 50), APPLE_BLUE, -1)
                    cv2.putText(preview_frame, "Confirm Registration", (20, 35), 
                                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
                    
                    # Bottom control bar
                    cv2.rectangle(preview_frame, (0, h+offset_y), (w, h+offset_y+50), (255, 255, 255), -1)
                    
                    # Action buttons
                    cv2.putText(preview_frame, "Press [Y] to Confirm", (20, h+offset_y+35), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_GREEN, 1)
                    cv2.putText(preview_frame, "Press [R] to Retake", (w//2-80, h+offset_y+35), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_BLUE, 1)
                    cv2.putText(preview_frame, "Press [Q] to Cancel", (w-220, h+offset_y+35), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, APPLE_RED, 1)
                    
                    # Show preview
                    cv2.imshow('Face Registration', preview_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    
                    # If 'y' pressed, confirm capture
                    if key == ord('y'):
                        capture_confirmed = True
                        break
                    
                    # If 'r' pressed, return to capture mode
                    elif key == ord('r'):
                        preview_mode = False
                        face_captured = False
                        print("Retaking image...")
                    
                    # If 'q' pressed, cancel registration
                    elif key == ord('q'):
                        print("Registration cancelled")
                        break
        
        finally:
            # Release camera
            cap.release()
            cv2.destroyAllWindows()
        
        # Process captured image
        if face_captured and capture_confirmed:
            success = self.encode_face(temp_image_path, name)
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
            return success
        
        # Clean up temp file if exists but registration was cancelled
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
        return False

    def recognize_faces(self, face_locations: List[Tuple[int, int, int, int]], 
                       frame: np.ndarray) -> List[Tuple[str, Tuple]]:
        """
        Recognize faces in the frame.
        
        Args:
            face_locations: List of face locations (top, right, bottom, left)
            frame: RGB image frame
            
        Returns:
            List of tuples (name, face_location)
        """
        if not face_locations:
            return []
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if face matches any known face
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    
            # Scale back face location (since we resized the frame)
            scaled_location = tuple(4 * coord for coord in face_location)
            results.append((name, scaled_location))
            
        return results