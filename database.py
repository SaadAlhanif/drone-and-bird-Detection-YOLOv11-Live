import sqlite3

DB_NAME = "detections.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        detected_at TEXT NOT NULL,
        image_path TEXT NOT NULL,
        source TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def insert_detection(object_type, confidence, detected_at, image_path, source):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO detections (object_type, confidence, detected_at, image_path, source)
    VALUES (?, ?, ?, ?, ?)
    """, (object_type, confidence, detected_at, image_path, source))

    conn.commit()
    conn.close()


def get_all_detections():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, object_type, confidence, detected_at, image_path, source
    FROM detections
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows
