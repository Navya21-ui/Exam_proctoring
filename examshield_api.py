import cv2
import datetime
import threading
from flask import Flask, jsonify
from flask_cors import CORS
import time
from collections import deque


from modules.face_recognition_module import load_known_faces, recognize_face
from modules.Liveness_module import LivenessDetector
from modules.behavior_module import BehaviorMonitor


app = Flask(__name__)
CORS(app) 


monitoring_state = {} 


monitoring_history = deque(maxlen=30) 


def webcam_monitor():
    
    global monitoring_state, monitoring_history

    
    liveness_detector = LivenessDetector()
    behavior_monitor = BehaviorMonitor()
    try:
        known_encodings, known_names = load_known_faces('examshield.db')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load faces from database. Error: {e}")
        monitoring_state["status"] = "Error: Database load failed."
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CRITICAL ERROR: Cannot open webcam.")
        monitoring_state["status"] = "Error: Cannot open webcam."
        return
        
    monitoring_state["status"] = "Running"
    print("✅ Webcam monitor thread started successfully.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_count, behavior_alert, landmarks = behavior_monitor.check_behavior(rgb_frame)

        attention_score = 0
        liveness_status = "N/A"
        recognized_student = "N/A"

        if face_count == 1:
            liveness_status, _ = liveness_detector.check_liveness(landmarks)
            small_frame_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            recognized_student = recognize_face(small_frame_rgb, known_encodings, known_names)
            
            if liveness_status == "Live" and behavior_alert == "None":
                attention_score = 100
            elif liveness_status == "Warning: Liveness Failed":
                attention_score = 25
            else:
                attention_score = 75

        elif face_count == 0:
            attention_score = 0
        else:
            recognized_student = "Multiple People"
            attention_score = 10

        timestamp = datetime.datetime.now()
        current_state = {
            "status": "Running",
            "recognized_student": recognized_student,
            "face_count": face_count,
            "behavior_alert": behavior_alert,
            "liveness_status": liveness_status,
            "attention_score": attention_score,
            "last_update": timestamp.isoformat(),
        }
        
        monitoring_state = current_state
        monitoring_history.append(current_state)

    cap.release()


@app.route("/")
def home():
    return jsonify({"message": "ExamShield Monitoring API is running."})

@app.route("/status")
def get_status():
    return jsonify(monitoring_state)

@app.route("/history")
def get_history():
    return jsonify(list(monitoring_history))


if __name__ == "__main__":
    print("Starting the ExamShield monitoring server...")
    monitor_thread = threading.Thread(target=webcam_monitor, daemon=True)
    monitor_thread.start()
    
    
    app.run(host="0.0.0.0", port=5000, debug=False)

