import cv2
import time

# --- Import all the same modules as your API ---
from modules.face_recognition_module import load_known_faces, recognize_face
from modules.Liveness_module import LivenessDetector
from modules.behavior_module import BehaviorMonitor

# ==============================================================================
# 1. INITIALIZATION
# ==============================================================================
print("🚀 Starting Visual Tester...")

# --- Initialize Modules (same as the API) ---
liveness_detector = LivenessDetector()
behavior_monitor = BehaviorMonitor()
try:
    print("🧠 Loading known faces from database...")
    known_encodings, known_names = load_known_faces('examshield.db')
    print(f"✅ Loaded {len(known_names)} known faces.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load faces from database. Error: {e}")
    exit()

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CRITICAL ERROR: Cannot open webcam.")
    exit()

print("✅ Webcam started. Press 'q' in the window to quit.")

# ==============================================================================
# 2. MAIN REAL-TIME LOOP
# ==============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert frame to RGB, as required by the modules
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Step 1: Run the Behavior Module (the "gatekeeper") ---
    face_count, behavior_alert, landmarks = behavior_monitor.check_behavior(rgb_frame)

    # --- Step 2: Run Liveness and Recognition based on behavior ---
    liveness_status = "N/A"
    recognized_student = "N/A"

    if face_count == 1:
        # If one face is found, run the other checks
        liveness_status, _ = liveness_detector.check_liveness(landmarks)
        
        # For recognition, we can use a resized frame to be faster
        small_frame_rgb = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        recognized_student = recognize_face(small_frame_rgb, known_encodings, known_names)

    elif face_count > 1:
        recognized_student = "Multiple People"

    # ==========================================================================
    # 3. VISUALIZATION (The "Mirror" Part)
    # This is where we draw the results directly onto the frame.
    # ==========================================================================
    
    # --- Define colors for status text ---
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    font = cv2.FONT_HERSHEY_DUPLEX

    # --- Create a status bar at the top ---
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (20, 20, 20), -1)

    # Display Recognized Student
    student_color = RED if recognized_student in ["Unrecognized", "Multiple People"] else GREEN
    cv2.putText(frame, f"Student: {recognized_student}", (10, 30), font, 0.8, student_color, 2)

    # Display Liveness Status
    liveness_color = RED if "Failed" in liveness_status else GREEN
    cv2.putText(frame, f"Liveness: {liveness_status}", (10, 60), font, 0.8, liveness_color, 2)

    # Display Behavior Alert
    alert_color = RED if "ALERT" in behavior_alert else GREEN
    cv2.putText(frame, f"Behavior: {behavior_alert}", (10, 90), font, 0.8, alert_color, 2)

    # --- Display the final frame in a pop-up window ---
    cv2.imshow("ExamShield - Live Backend Visualization", frame)

    # --- Quit condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================================================================
# 4. CLEANUP
# ==============================================================================
print("🛑 Shutting down visual tester.")
cap.release()
cv2.destroyAllWindows()



    
