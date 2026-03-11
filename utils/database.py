# utils/database.py
import sqlite3
from datetime import datetime
import json
from config import Config

class ChatDatabase:
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                intent TEXT,
                confidence REAL,
                sentiment TEXT,
                entities TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_conversations INTEGER DEFAULT 0,
                unique_sessions INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                positive_sentiment INTEGER DEFAULT 0,
                negative_sentiment INTEGER DEFAULT 0,
                neutral_sentiment INTEGER DEFAULT 0
            )
        ''')
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conversation_id INTEGER,
                rating INTEGER,
                feedback_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id, user_message, bot_response, 
                          intent, confidence, sentiment, entities):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, user_message, bot_response, intent, confidence, sentiment, entities)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, intent, confidence, 
              sentiment, json.dumps(entities)))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id, limit=10):
        """Get conversation history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_message, bot_response, timestamp 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_analytics_data(self, days=30):
        """Get analytics data for dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_messages,
                COUNT(DISTINCT session_id) as unique_sessions,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM conversations
            WHERE timestamp >= DATE('now', ?)
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', (f'-{days} days',))
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_intent_distribution(self):
        """Get intent distribution for analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT intent, COUNT(*) as count
            FROM conversations
            WHERE intent IS NOT NULL
            GROUP BY intent
            ORDER BY count DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_recent_conversations(self, limit=50):
        """Get recent conversations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, user_message, bot_response, intent, 
                   confidence, sentiment, timestamp
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        return results