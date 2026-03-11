# config.py
import os

class Config:
    # Database
    DATABASE_PATH = "data/chat_history.db"
    
    # NLP Settings
    SPACY_MODEL = "en_core_web_sm"
    CONFIDENCE_THRESHOLD = 0.6
    
    # Chatbot Settings
    MAX_CONTEXT_LENGTH = 5
    DEFAULT_RESPONSE = "I'm sorry, I didn't understand that. Could you please rephrase?"
    
    # Analytics
    SESSION_TIMEOUT = 30  # minutes