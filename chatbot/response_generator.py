import random
import json


class ResponseGenerator:
    def __init__(self, intents_file='data/intents.json'):
        self.intents_file = intents_file
        self.intents_data = self.load_intents()
        self.fallback_responses = [
            "I am not sure I understand. Could you please rephrase that?",
            "I did not quite catch that. Can you try saying it differently?",
            "I am still learning! Could you explain what you mean?",
            "Sorry, I could not understand. Can you provide more details?"
        ]

    def load_intents(self):
        try:
            with open(self.intents_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"intents": []}

    def generate_response(self, intent, confidence, user_message, context=None):
        try:
            if confidence < 0.15:
                return random.choice(self.fallback_responses)

            response = self.find_response(intent)

            if response is None:
                return random.choice(self.fallback_responses)

            return str(response)
        except Exception:
            return random.choice(self.fallback_responses)

    def find_response(self, intent):
        try:
            intents_list = self.intents_data.get('intents', [])
            for item in intents_list:
                tag = item.get('tag', '')
                if tag == intent:
                    responses = item.get('responses', [])
                    if len(responses) > 0:
                        return str(random.choice(responses))
            return None
        except Exception:
            return None

    def get_responses_for_intent(self, intent):
        try:
            intents_list = self.intents_data.get('intents', [])
            for item in intents_list:
                if item.get('tag', '') == intent:
                    return item.get('responses', [])
            return []
        except Exception:
            return []
