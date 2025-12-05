import sqlite3

def convert_to_binary_data(filename):
    """Reads a file and converts it into a binary blob."""
    with open(filename, 'rb') as file:
        blob_data = file.read()
    return blob_data

def insert_student_image(name, photo_filename):
    """Inserts a student's name and photo into the database."""
    try:
        conn = sqlite3.connect('examshield.db')
        cursor = conn.cursor()
        print(f"🔗 Connected to database. Inserting {name}...")
        student_photo = convert_to_binary_data(photo_filename)
        sql_insert_query = "INSERT OR IGNORE INTO students (name, image) VALUES (?, ?)"
        data_tuple = (name, student_photo)
        cursor.execute(sql_insert_query, data_tuple)
        conn.commit()
        print(f"✅ Image for {name} has been inserted successfully.")
    except sqlite3.Error as error:
        print(f"❌ Failed to insert data into sqlite table: {error}")
    finally:
        if conn:
            conn.close()
            print("🚪 The SQLite connection is closed.")


insert_student_image('Navya', 'Navya_face.jpg')
#insert_student_image('Manshi', 'Manshi_face.jpg')
insert_student_image('Susheela', 'Mumma_face.jpg')
