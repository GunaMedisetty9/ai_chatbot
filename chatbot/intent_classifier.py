# chatbot/intent_classifier.py
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from chatbot.nlp_engine import NLPEngine
import pickle
import os

class IntentClassifier:
    def __init__(self, intents_file='data/intents.json'):
        self.nlp = NLPEngine()
        self.intents_file = intents_file
        self.intents_data = self.load_intents()
        self.model = None
        self.vectorizer = None
        self.classes = []
        self.train_model()
    
    def load_intents(self):
        """Load intents from JSON file"""
        try:
            with open(self.intents_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default intents if file not found
            return self.get_default_intents()
    
    def get_default_intents(self):
        """Return default intents structure"""
        return {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["hi", "hello", "hey", "good morning"],
                    "responses": ["Hello! How can I help you?"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["bye", "goodbye", "see you"],
                    "responses": ["Goodbye! Have a great day!"]
                },
                {
                    "tag": "unknown",
                    "patterns": [],
                    "responses": ["I'm not sure I understand. Could you rephrase?"]
                }
            ]
        }
    
    def train_model(self):
        """Train the intent classification model"""
        training_sentences = []
        training_labels = []
        self.classes = []
        
        for intent in self.intents_data['intents']:
            tag = intent['tag']
            if tag not in self.classes:
                self.classes.append(tag)
            
            for pattern in intent['patterns']:
                # Preprocess and add to training data
                processed = ' '.join(self.nlp.preprocess(pattern))
                training_sentences.append(processed)
                training_labels.append(tag)
        
        if not training_sentences:
            return
        
        # Create and train pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', MultinomialNB())
        ])
        
        self.model.fit(training_sentences, training_labels)
    
    def predict(self, text):
        """Predict intent for given text"""
        if not self.model:
            return "unknown", 0.0
        
        processed_text = ' '.join(self.nlp.preprocess(text))
        
        # Get prediction and probability
        prediction = self.model.predict([processed_text])[0]
        probabilities = self.model.predict_proba([processed_text])[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
    def get_all_intents(self):
        """Return all available intents"""
        return [intent['tag'] for intent in self.intents_data['intents']]
    
    def add_intent(self, tag, patterns, responses):
        """Add a new intent dynamically"""
        new_intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }
        self.intents_data['intents'].append(new_intent)
        self.save_intents()
        self.train_model()  # Retrain model
    
    def save_intents(self):
        """Save intents to JSON file"""
        os.makedirs(os.path.dirname(self.intents_file), exist_ok=True)
        with open(self.intents_file, 'w') as f:
            json.dump(self.intents_data, f, indent=4)