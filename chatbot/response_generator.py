import random
import json
from chatbot.sentiment_analyzer import SentimentAnalyzer


class ResponseGenerator:
    def __init__(self, intents_file='data/intents.json'):
        self.intents_file = intents_file
        self.intents_data = self.load_intents()
        self.sentiment_analyzer = SentimentAnalyzer()

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

        self.fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "I didn't quite catch that. Can you try saying it differently?",
            "I'm still learning! Could you explain what you mean?",
            "Sorry, I couldn't understand. Can you provide more details?"
        ]

    def load_intents(self):
        try:
            with open(self.intents_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"intents": []}

    def generate_response(self, intent, confidence, user_message, context=None):
        """Generate response"""

        sentiment_result = self.sentiment_analyzer.analyze(user_message)
        suggested_tone = self.sentiment_analyzer.adjust_response_tone(sentiment_result)

        # Very low threshold since keyword matching ensures good intent
        if confidence < 0.15:
            return self._get_fallback_response(suggested_tone)

        response = self._get_intent_response(intent)

        if not response:
            return self._get_fallback_response(suggested_tone)

        # Add tone prefix for negative sentiment
        if suggested_tone == 'empathetic' and sentiment_result['sentiment'] == 'negative':
            prefix = random.choice(self.tone_prefixes['empathetic'])
            response = prefix + response

        # Personalize only if context exists and is valid
        if context and isinstance(context, dict):
            response = self._personalize_response(response, context)

        return response

    def _get_intent_response(self, intent):
        for intent_data in self.intents_data['intents']:
            if intent_data['tag'] == intent:
                responses = intent_data.get('responses', [])
                if responses:
                    return random.choice(responses)
        return None

    def _get_fallback_response(self, tone='neutral'):
        base = random.choice(self.fallback_responses)
        if tone == 'empathetic':
            prefix = random.choice(self.tone_prefixes['empathetic'])
            return prefix + base
        return base

    def _personalize_response(self, response, context):
        """Safe personalization with None checks"""
        try:
            user_name = context.get('user_name', None)
            if user_name and isinstance(user_name, str):
                response = response.replace('{user_name}', user_name)
            else:
                response = response.replace('{user_name}', '')

            last_topic = context.get('last_topic', None)
            if last_topic and isinstance(last_topic, str):
                response = response.replace('{topic}', last_topic)
            else:
                response = response.replace('{topic}', '')
        except Exception:
            pass

        return response

    def get_responses_for_intent(self, intent):
        for intent_data in self.intents_data['intents']:
            if intent_data['tag'] == intent:
                return intent_data.get('responses', [])
        return []
