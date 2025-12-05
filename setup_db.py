import sqlite3

conn = sqlite3.connect('examshield.db')
cursor = conn.cursor()
create_table_query = '''
CREATE TABLE IF NOT EXISTS students (
    name TEXT PRIMARY KEY,
    image BLOB NOT NULL
);
'''


cursor.execute(create_table_query)

print("✅ Database 'examshield.db' and table 'students' are ready.")

conn.commit()
conn.close()