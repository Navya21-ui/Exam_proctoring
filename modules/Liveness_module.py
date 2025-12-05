import mediapipe as mp
import math

class LivenessDetector:
    def __init__(self):
       
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.BLINK_THRESHOLD = 0.25
        self.NO_BLINK_FRAME_LIMIT = 7 * 20 
        
        
        self.frames_without_blink = 0

    def _euclidean_distance(self, point1, point2):
        """Helper function to calculate distance between two landmark points."""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def _calculate_ear(self, landmarks, eye_indices):
        """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
       
        points = [landmarks[i] for i in eye_indices]
        
        
        v1 = self._euclidean_distance(points[1], points[5])
        v2 = self._euclidean_distance(points[2], points[4])
        
        
        h = self._euclidean_distance(points[0], points[3])
        
        
        if h == 0: return 0.0
        
        
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def check_liveness(self, landmarks):
        
        if not landmarks or len(landmarks) < 400: 
            return "N/A", self.frames_without_blink

        
        left_ear = self._calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        
        status = "Live" 
        
        if avg_ear < self.BLINK_THRESHOLD:
            
            self.frames_without_blink = 0
        else:
            
            self.frames_without_blink += 1

        
        if self.frames_without_blink > self.NO_BLINK_FRAME_LIMIT:
            status = "Warning: Liveness Failed"
        
        return status, self.frames_without_blink

