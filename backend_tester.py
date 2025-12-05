import cv2
import time

from modules.face_recognition_module import load_known_faces, recognize_face
from modules.Liveness_module import LivenessDetector
from modules.behavior_module import BehaviorMonitor

print("🚀 Starting Visual Tester...")

liveness_detector = LivenessDetector()
behavior_monitor = BehaviorMonitor()
try:
    print("🧠 Loading known faces from database...")
    known_encodings, known_names = load_known_faces('examshield.db')
    print(f"✅ Loaded {len(known_names)} known faces.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load faces from database. Error: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CRITICAL ERROR: Cannot open webcam.")
    exit()

print("✅ Webcam started. Press 'q' in the window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_count, behavior_alert, landmarks = behavior_monitor.check_behavior(rgb_frame)

    liveness_status = "N/A"
    recognized_student = "N/A"

    if face_count == 1:
        liveness_status, _ = liveness_detector.check_liveness(landmarks)
        small_frame_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        recognized_student = recognize_face(small_frame_rgb, known_encodings, known_names)
    elif face_count > 1:
        recognized_student = "Multiple People"

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    font = cv2.FONT_HERSHEY_DUPLEX

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (20, 20, 20), -1)

    student_color = RED if recognized_student in ["Unrecognized", "Multiple People"] else GREEN
    cv2.putText(frame, f"Student: {recognized_student}", (10, 30), font, 0.8, student_color, 2)

    liveness_color = RED if "Failed" in liveness_status else GREEN
    cv2.putText(frame, f"Liveness: {liveness_status}", (10, 60), font, 0.8, liveness_color, 2)

    alert_color = RED if "ALERT" in behavior_alert else GREEN
    cv2.putText(frame, f"Behavior: {behavior_alert}", (10, 90), font, 0.8, alert_color, 2)

    cv2.imshow("ExamShield - Live Backend Visualization", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("🛑 Shutting down visual tester.")
cap.release()
cv2.destroyAllWindows()
