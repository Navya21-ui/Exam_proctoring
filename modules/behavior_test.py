import cv2
import time


from modules.behavior_module import BehaviorMonitor


monitor = BehaviorMonitor()
cap = cv2.VideoCapture(0)

print(" Starting Behavior Monitor Test. Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    face_count, alert, landmarks = monitor.check_behavior(rgb_frame)

    
    print(f"Faces Detected: {face_count}, Alert Status: {alert}")

    
    cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Alert: {alert}", (10, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255) if "ALERT" in alert else (0, 255, 0), 2)

    cv2.imshow("Behavior Module Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
