from datetime import datetime
import uuid
from chatbot.nlp_engine import NLPEngine
from chatbot.intent_classifier import IntentClassifier
from chatbot.entity_extractor import EntityExtractor
from chatbot.sentiment_analyzer import SentimentAnalyzer
from chatbot.response_generator import ResponseGenerator
from utils.database import ChatDatabase

class ConversationManager:
    def __init__(self):
        self.nlp = NLPEngine()
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_generator = ResponseGenerator()
        self.database = ChatDatabase()
        
        # Context memory for conversations
        self.context = {}
        self.max_context_length = 5
    
    def create_session(self):
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.context[session_id] = {
            'history': [],
            'entities': {},
            'last_intent': None,
            'user_name': None,
            'created_at': datetime.now()
        }
        return session_id
    
    def process_message(self, session_id, user_message):
        """Process a user message and generate response"""
        
        # Ensure session exists
        if session_id not in self.context:
            self.context[session_id] = {
                'history': [],
                'entities': {},
                'last_intent': None,
                'user_name': None,
                'created_at': datetime.now()
            }
        
        # 1. Intent Classification
        intent, confidence = self.intent_classifier.predict(user_message)
        
        # 2. Entity Extraction
        entities = self.entity_extractor.extract_entities(user_message)
        
        # 3. Sentiment Analysis
        sentiment_result = self.sentiment_analyzer.analyze(user_message)
        
        # 4. Update context
        self._update_context(session_id, user_message, intent, entities)
        
        # 5. Generate Response
        response = self.response_generator.generate_response(
            intent=intent,
            confidence=confidence,
            user_message=user_message,
            context=self.context[session_id]
        )
        
        # 6. Save to database
        self.database.save_conversation(
            session_id=session_id,
            user_message=user_message,
            bot_response=response,
            intent=intent,
            confidence=confidence,
            sentiment=sentiment_result['sentiment'],
            entities=entities
        )
        
        # 7. Prepare response data
        response_data = {
            'response': response,
            'intent': intent,
            'confidence': round(confidence, 3),
            'sentiment': sentiment_result,
            'entities': entities,
            'session_id': session_id
        }
        
        return response_data
    
    def _update_context(self, session_id, message, intent, entities):
        """Update conversation context"""
        context = self.context[session_id]
        
        # Add to history
        context['history'].append({
            'message': message,
            'intent': intent,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(context['history']) > self.max_context_length:
            context['history'] = context['history'][-self.max_context_length:]
        
        # Update last intent
        context['last_intent'] = intent
        
        # Merge entities
        for entity_type, values in entities.items():
            if entity_type not in context['entities']:
                context['entities'][entity_type] = []
            context['entities'][entity_type].extend(values)
    
    def get_context(self, session_id):
        """Get current context for a session"""
        return self.context.get(session_id, {})
    
    def clear_context(self, session_id):
        """Clear context for a session"""
        if session_id in self.context:
            del self.context[session_id]
    
    def get_conversation_history(self, session_id):
        """Get conversation history from database"""
        return self.database.get_conversation_history(session_id)