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
        image = face_recognition.load_image_file(image_path)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return False
        
        # Use the first face found
        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        
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
    Register a face using live camera feed with preview functionality.
    
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
    temp_image_path = os.path.join(os.path.dirname(self.encodings_dir), "temp", f"{name}_{time.time()}.jpg")
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
    
    captured_frame = None
    
    try:
        # State management
        preview_mode = False
        
        while True:
            if not preview_mode:
                # Camera capture mode
                ret, frame = cap.read()
                
                if not ret:
                    continue
                    
                # Detect faces in current frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                # Draw rectangle around faces
                for top, right, bottom, left in face_locations:
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw box around face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Display status message on frame
                cv2.putText(frame, "Press 'c' to capture, 'q' to cancel", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Registration - Face Capture', frame)
                
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
                preview_frame = captured_frame.copy()
                
                # Add instructions for confirmation
                cv2.putText(preview_frame, "Press 'y' to confirm or 'r' to retake", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show captured image
                cv2.imshow('Registration - Face Capture', preview_frame)
                
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