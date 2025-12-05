import sqlite3
import numpy as np
import cv2
import face_recognition

def load_known_faces(db_path='examshield.db'):
    
    known_face_encodings = []
    known_face_names = []
    print(" Loading known faces from database ")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, image FROM students")
        rows = cursor.fetchall()
        for name, img_blob in rows:
            nparr = np.frombuffer(img_blob, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
        conn.close()
        print(f" Loaded {len(known_face_names)} known faces.")
    except Exception as e:
        print(f" Database error in face recognition module: {e}")
    return known_face_encodings, known_face_names

def recognize_face(rgb_frame, known_encodings, known_names):
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=1.5)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
        return name 
    
    return "Unknown" 
