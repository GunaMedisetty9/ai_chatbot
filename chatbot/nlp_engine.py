# chatbot/nlp_engine.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class NLPEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep some important words that are usually stopwords
        self.important_words = {'not', 'no', 'help', 'can', 'what', 'how', 'why', 'when', 'where'}
        self.stop_words = self.stop_words - self.important_words
    
    def preprocess(self, text):
        """
        Preprocess text: lowercase, remove punctuation, tokenize, lemmatize
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\?\!]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
        
        return processed_tokens
    
    def get_raw_tokens(self, text):
        """Get raw tokens without heavy preprocessing"""
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens
    
    def extract_keywords(self, text, top_n=5):
        """Extract top keywords from text"""
        tokens = self.preprocess(text)
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]
    
    def calculate_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts"""
        tokens1 = set(self.preprocess(text1))
        tokens2 = set(self.preprocess(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)