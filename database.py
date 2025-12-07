import sqlite3
from datetime import datetime

DB = "sentiments.db"

def _conn():
    return sqlite3.connect(DB)

def init_db():
    conn = sqlite3.connect('sentiment_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        sentiment TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_record(text, sentiment):
    conn = sqlite3.connect('sentiment_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO sentiment_history (text, sentiment, timestamp)
    VALUES (?, ?, ?)
    ''', (text, sentiment, datetime.now()))
    
    conn.commit()
    conn.close()
def get_by_page(limit=5, offset=0):
    conn = sqlite3.connect('sentiment_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, text, sentiment, timestamp
    FROM sentiment_history
    ORDER BY id ASC
    LIMIT ? OFFSET ?
    ''', (limit, offset))
    
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_latest(limit=50, offset=0):
    conn = sqlite3.connect('sentiment_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, text, sentiment, timestamp
    FROM sentiment_history
    ORDER BY timestamp DESC
    LIMIT ? OFFSET ?
    ''', (limit, offset))
    
    rows = cursor.fetchall()
    conn.close()
    
    return rows

def get_total_count():
    conn = sqlite3.connect('sentiment_history.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM sentiment_history')
    count = cursor.fetchone()[0]
    
    conn.close()
    return count
