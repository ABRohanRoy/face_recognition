import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Dict, Optional

class FaceDetector:
    def __init__(self, detection_method='hog'):
        """
        Initialize face detector.
        
        Args:
            detection_method: 'hog' (faster) or 'cnn' (more accurate, requires GPU)
        """
        self.detection_method = detection_method
        
    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Image frame from video stream
            
        Returns:
            Tuple of (face_locations, small_frame)
        """
        # Resize frame to 1/4 size for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert BGR to RGB
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            rgb_small_frame, model=self.detection_method
        )
        
        return face_locations, rgb_small_frame
