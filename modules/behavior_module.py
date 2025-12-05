import mediapipe as mp
import time

class BehaviorMonitor:
    def __init__(self):
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5, 
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        
        self.NO_FACE_ALERT_SECONDS = 5
        self.MULTI_FACE_ALERT_SECONDS = 5
        
        
        self.no_face_start_time = None
        self.multi_face_start_time = None

    def check_behavior(self, rgb_frame):
        
        
        results = self.face_mesh.process(rgb_frame)
        
        face_count = 0
        landmarks = None
        alert = "None" 

        
        if results.multi_face_landmarks:
            face_count = len(results.multi_face_landmarks)
            
            landmarks = results.multi_face_landmarks[0].landmark

        
        if face_count == 0:
            if self.no_face_start_time is None:
                
                self.no_face_start_time = time.time()
            elif time.time() - self.no_face_start_time > self.NO_FACE_ALERT_SECONDS:
                
                alert = "ALERT: No face detected!"
        else:
            
            self.no_face_start_time = None

        
        if face_count > 1:
            if self.multi_face_start_time is None:
                
                self.multi_face_start_time = time.time()
            elif time.time() - self.multi_face_start_time > self.MULTI_FACE_ALERT_SECONDS:
                
                alert = "ALERT: Multiple faces detected!"
        else:
            
            self.multi_face_start_time = None

        
        return face_count, alert, landmarks

