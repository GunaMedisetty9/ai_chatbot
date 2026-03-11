import random
import json


class ResponseGenerator:
    def __init__(self, intents_file='data/intents.json'):
        self.intents_file = intents_file
        self.intents_data = self.load_intents()
        self.fallback = [
            "I am not sure I understand. Could you please rephrase that?",
            "I did not quite catch that. Can you try saying it differently?",
            "I am still learning! Could you explain what you mean?",
            "Sorry, I could not understand. Can you provide more details?"
        ]

    def load_intents(self):
        try:
            with open(self.intents_file, 'r') as f:
                data = json.load(f)
                return data
        except Exception:
            return {"intents": []}

    def generate_response(self, intent, confidence, user_message, context=None):
        if confidence < 0.15:
            return random.choice(self.fallback)
        answer = self.find_answer(intent)
        if answer is None:
            return random.choice(self.fallback)
        return answer

    def find_answer(self, intent):
        try:
            for item in self.intents_data.get('intents', []):
                if item.get('tag', '') == intent:
                    r = item.get('responses', [])
                    if len(r) > 0:
                        picked = random.choice(r)
                        return str(picked)
            return None
        except Exception:
            return None

    def get_responses_for_intent(self, intent):
        try:
            for item in self.intents_data.get('intents', []):
                if item.get('tag', '') == intent:
                    return item.get('responses', [])
            return []
        except Exception:
            return []
