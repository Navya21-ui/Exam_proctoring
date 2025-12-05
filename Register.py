import cv2
import sqlite3
import numpy as np

DB_NAME = 'examshield.db'

def insert_student_image(name, image_frame):
    """Inserts a student's name and photo (as a numpy frame) into the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        
        is_success, buffer = cv2.imencode(".jpg", image_frame)
        if not is_success:
            print("❌ Failed to encode image.")
            return

        
        student_photo_blob = buffer.tobytes()

        #
        sql_insert_query = "INSERT OR IGNORE INTO students (name, image) VALUES (?, ?)"
        data_tuple = (name, student_photo_blob)
        
        cursor.execute(sql_insert_query, data_tuple)
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"✅ User '{name}' has been registered successfully.")
        else:
            print(f"⚠️ User '{name}' already exists in the database.")

    except sqlite3.Error as error:
        print(f"❌ Failed to insert data into sqlite table: {error}")
    finally:
        if conn:
            conn.close()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


student_name = input("Enter the new student's name and press Enter: ")

if not student_name:
    print("Name cannot be empty. Exiting.")
else:
    cap = cv2.VideoCapture(0)
    print("📷 Webcam opened. Align face in the frame.")
    print("Press 's' to save the image. Press 'q' to quit without saving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        
        cv2.putText(frame, "Align face and press 's' to save", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Register New Student", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Registration cancelled.")
            break
        
        if key == ord('s'):
            if len(faces) > 0:
                print("Capturing image...")
                insert_student_image(student_name, frame)
                break
            else:
                print("No face detected! Please get closer to the camera.")

    cap.release()
    cv2.destroyAllWindows()