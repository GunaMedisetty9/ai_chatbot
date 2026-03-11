import random
import json


class ResponseGenerator:
    def __init__(self, intents_file='data/intents.json'):
        self.intents_file = intents_file
        self.intents_data = self.load_intents()

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
        except Exception:
            return {"intents": []}

    def generate_response(self, intent, confidence, user_message, context=None):
        """Generate response based on intent"""
        try:
            # Low confidence = fallback
            if confidence < 0.15:
                return random.choice(self.fallback_responses)

            # Find response for intent
            response = self._get_intent_response(intent)

            # No response found = fallback
            if response is None or response == "":
                return random.choice(self.fallback_responses)

            return str(response)

        except Exception:
            return random.choice(self.fallback_responses)

    def _get_intent_response(self, intent):
        """Get a random response for the given intent"""
        try:
            for intent_data in self.intents_data.get('intents', []):
                if intent_data.get('tag', '') == intent:
                    responses = intent_data.get('responses', [])
                    if responses and len(responses) > 0:
                        return str(random.choice(responses))
            return None
        except Exception:
            return None

    def get_responses_for_intent(self, intent):
        try:
            for intent_data in self.intents_data.get('intents', []):
                if intent_data.get('tag', '') == intent:
                    return intent_data.get('responses', [])
            return []
        except Exception:
            return []
