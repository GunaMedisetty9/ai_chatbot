from datetime import datetime
import uuid
import random
import json
from chatbot.nlp_engine import NLPEngine
from chatbot.intent_classifier import IntentClassifier
from chatbot.entity_extractor import EntityExtractor
from chatbot.sentiment_analyzer import SentimentAnalyzer
from utils.database import ChatDatabase


class ConversationManager:
    def __init__(self):
        self.nlp = NLPEngine()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.database = ChatDatabase()
        self.context = {}
        self.intents_data = self.load_intents()
        self.fallback = [
            "I am not sure I understand. Could you rephrase?",
            "I did not catch that. Try saying it differently?",
            "I am still learning! Could you explain?",
            "Sorry, can you provide more details?"
        ]

    def load_intents(self):
        try:
            with open('data/intents.json', 'r') as f:
                return json.load(f)
        except Exception:
            return {"intents": []}

    def get_response(self, intent, confidence):
        if confidence < 0.15:
            return random.choice(self.fallback)
        try:
            for item in self.intents_data.get('intents', []):
                if item.get('tag', '') == intent:
                    responses = item.get('responses', [])
                    if len(responses) > 0:
                        return str(random.choice(responses))
            return random.choice(self.fallback)
        except Exception:
            return random.choice(self.fallback)

    def process_message(self, session_id, user_message):
        try:
            if session_id not in self.context:
                self.context[session_id] = {
                    'history': [],
                    'last_intent': None
                }

            try:
                intent, confidence = self.intent_classifier.predict(user_message)
            except Exception:
                intent = "unknown"
                confidence = 0.0

            try:
                entities = self.entity_extractor.extract_entities(user_message)
            except Exception:
                entities = {}

            try:
                sentiment_result = self.sentiment_analyzer.analyze(user_message)
            except Exception:
                sentiment_result = {
                    'sentiment': 'neutral',
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'confidence': 0.0,
                    'emotions': ['neutral'],
                    'has_intensifier': False
                }

            response = self.get_response(intent, confidence)

            try:
                self.database.save_conversation(
                    session_id,
                    user_message,
                    response,
                    str(intent),
                    float(confidence),
                    str(sentiment_result.get('sentiment', 'neutral')),
                    entities
                )
            except Exception:
                pass

            try:
                self.context[session_id]['history'].append({
                    'message': user_message,
                    'intent': intent
                })
                self.context[session_id]['last_intent'] = intent
                if len(self.context[session_id]['history']) > 5:
                    self.context[session_id]['history'] = self.context[session_id]['history'][-5:]
            except Exception:
                pass

            return {
                'response': response,
                'intent': str(intent),
                'confidence': round(float(confidence), 3),
                'sentiment': sentiment_result,
                'entities': entities,
                'session_id': session_id
            }

        except Exception:
            return {
                'response': 'I am here to help! Please try again.',
                'intent': 'unknown',
                'confidence': 0.0,
                'sentiment': {
                    'sentiment': 'neutral',
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'confidence': 0.0,
                    'emotions': ['neutral'],
                    'has_intensifier': False
                },
                'entities': {},
                'session_id': session_id
            }

    def get_context(self, session_id):
        return self.context.get(session_id, {})

    def clear_context(self, session_id):
        if session_id in self.context:
            del self.context[session_id]
