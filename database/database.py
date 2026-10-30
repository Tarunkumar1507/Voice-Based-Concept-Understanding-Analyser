import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "vbcua.db")

def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Establishes and returns a database connection.
    Automatically creates parent directory if it does not exist.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    return sqlite3.connect(db_path)

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Initializes the SQLite database schema if the table does not exist.
    """
    query = """
    CREATE TABLE IF NOT EXISTS student_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        topic TEXT NOT NULL,
        transcript TEXT,
        semantic_score REAL,
        speech_score REAL,
        overall_score REAL,
        feedback TEXT,
        created_at TEXT NOT NULL
    );
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        raise

def save_report(
    student_name: str,
    topic: str,
    transcript: str,
    semantic_score: float,
    speech_score: float,
    overall_score: float,
    feedback: str,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Saves a student evaluation report to the database.
    Returns the ID of the inserted record.
    """
    query = """
    INSERT INTO student_reports (
        student_name, topic, transcript, semantic_score, speech_score, overall_score, feedback, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                student_name,
                topic,
                transcript,
                semantic_score,
                speech_score,
                overall_score,
                feedback,
                created_at
            ))
            conn.commit()
            return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"Error saving report to database: {e}")
        raise

def get_history(db_path: str = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """
    Retrieves all student reports from the database ordered by creation date descending.
    """
    query = """
    SELECT id, student_name, topic, transcript, semantic_score, speech_score, overall_score, feedback, created_at
    FROM student_reports
    ORDER BY created_at DESC;
    """
    reports = []
    try:
        with get_connection(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            for row in rows:
                reports.append(dict(row))
    except sqlite3.Error as e:
        print(f"Error fetching report history: {e}")
    return reports

def delete_report(report_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Deletes a specific student report by ID.
    Returns True if successfully deleted, False otherwise.
    """
    query = "DELETE FROM student_reports WHERE id = ?;"
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (report_id,))
            conn.commit()
            return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Error deleting report with ID {report_id}: {e}")
        return False

def clear_history(db_path: str = DEFAULT_DB_PATH) -> bool:
    """
    Clears all evaluation records in the student_reports table.
    """
    query = "DELETE FROM student_reports;"
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            return True
    except sqlite3.Error as e:
        print(f"Error clearing history: {e}")
        return False
