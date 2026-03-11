# chatbot/response_generator.py
import random
import json
from chatbot.sentiment_analyzer import SentimentAnalyzer

class ResponseGenerator:
    def __init__(self, intents_file='data/intents.json'):
        self.intents_file = intents_file
        self.intents_data = self.load_intents()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Response templates for different tones
        self.tone_prefixes = {
            'empathetic': [
                "I understand your concern. ",
                "I'm sorry to hear that. ",
                "I can see this is frustrating. "
            ],
            'enthusiastic': [
                "That's great! ",
                "Wonderful! ",
                "Excellent! "
            ],
            'supportive': [
                "I'm here to help. ",
                "Let me assist you. ",
                "I'll do my best to help. "
            ],
            'neutral': [""]
        }
        
        # Fallback responses
        self.fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "I didn't quite catch that. Can you try saying it differently?",
            "I'm still learning! Could you explain what you mean?",
            "Sorry, I couldn't understand. Can you provide more details?"
        ]
    
    def load_intents(self):
        """Load intents from JSON file"""
        try:
            with open(self.intents_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"intents": []}
    
    def generate_response(self, intent, confidence, user_message, context=None):
        """Generate appropriate response based on intent and context"""
        
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze(user_message)
        suggested_tone = self.sentiment_analyzer.adjust_response_tone(sentiment_result)
        
        # Check confidence threshold
        if confidence < 0.4:
            return self._get_fallback_response(suggested_tone)
        
        # Find matching intent
        response = self._get_intent_response(intent)
        
        if not response:
            return self._get_fallback_response(suggested_tone)
        
        # Add tone prefix if needed
        if suggested_tone != 'neutral' and sentiment_result['sentiment'] == 'negative':
            prefix = random.choice(self.tone_prefixes[suggested_tone])
            response = prefix + response
        
        # Personalize response based on context
        if context:
            response = self._personalize_response(response, context)
        
        return response
    
    def _get_intent_response(self, intent):
        """Get a random response for the given intent"""
        for intent_data in self.intents_data['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data.get('responses', [])
                if responses:
                    return random.choice(responses)
        return None
    
    def _get_fallback_response(self, tone='neutral'):
        """Get a fallback response with appropriate tone"""
        base_response = random.choice(self.fallback_responses)
        
        if tone == 'empathetic':
            prefix = random.choice(self.tone_prefixes['empathetic'])
            return prefix + base_response
        
        return base_response
    
    def _personalize_response(self, response, context):
        """Personalize response based on context"""
        # Replace placeholders with context values
        if 'user_name' in context:
            response = response.replace('{user_name}', context['user_name'])
        
        if 'last_topic' in context:
            response = response.replace('{topic}', context['last_topic'])
        
        return response
    
    def get_responses_for_intent(self, intent):
        """Get all possible responses for an intent"""
        for intent_data in self.intents_data['intents']:
            if intent_data['tag'] == intent:
                return intent_data.get('responses', [])
        return []
    
    def add_response(self, intent, response):
        """Add a new response to an existing intent"""
        for intent_data in self.intents_data['intents']:
            if intent_data['tag'] == intent:
                if 'responses' not in intent_data:
                    intent_data['responses'] = []
                intent_data['responses'].append(response)
                self._save_intents()
                return True
        return False
    
    def _save_intents(self):
        """Save intents to file"""
        with open(self.intents_file, 'w') as f:
            json.dump(self.intents_data, f, indent=4)