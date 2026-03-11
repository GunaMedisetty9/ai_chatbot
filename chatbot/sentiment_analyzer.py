from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        # Emotion keywords for enhanced detection
        self.emotion_keywords = {
            'happy': ['happy', 'glad', 'joyful', 'excited', 'delighted', 'pleased', 'great', 'awesome', 'amazing', 'wonderful', 'love', 'fantastic'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'disappointed', 'upset', 'sorry', 'miss'],
            'angry': ['angry', 'furious', 'mad', 'annoyed', 'frustrated', 'irritated', 'hate', 'terrible', 'worst'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
            'surprised': ['surprised', 'amazed', 'shocked', 'astonished', 'wow'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'average']
        }
        
        self.intensifiers = ['very', 'really', 'extremely', 'absolutely', 'totally', 'completely', 'so']
        self.negations = ['not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing', "don't", "doesn't", "didn't"]
    
    def analyze(self, text):
        """Perform complete sentiment analysis"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        emotions = self._detect_emotions(text.lower())
        has_intensifier = self._check_intensifiers(text.lower())
        
        confidence = abs(polarity)
        if has_intensifier:
            confidence = min(confidence * 1.2, 1.0)
        
        return {
            'sentiment': sentiment,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3),
            'confidence': round(confidence, 3),
            'emotions': emotions,
            'has_intensifier': has_intensifier
        }
    
    def _detect_emotions(self, text):
        """Detect specific emotions in text"""
        detected_emotions = []
        
        has_negation = any(neg in text for neg in self.negations)
        
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    if has_negation and emotion in ['happy', 'sad', 'angry']:
                        opposite = {'happy': 'sad', 'sad': 'happy', 'angry': 'calm'}
                        detected_emotions.append(opposite.get(emotion, emotion))
                    else:
                        detected_emotions.append(emotion)
                    break
        
        return list(set(detected_emotions)) if detected_emotions else ['neutral']
    
    def _check_intensifiers(self, text):
        """Check if text contains intensifiers"""
        words = text.split()
        return any(intensifier in words for intensifier in self.intensifiers)
    
    def get_sentiment_label(self, text):
        """Get just the sentiment label"""
        analysis = self.analyze(text)
        return analysis['sentiment']
    
    def get_emotion(self, text):
        """Get the primary emotion"""
        analysis = self.analyze(text)
        emotions = analysis['emotions']
        return emotions[0] if emotions else 'neutral'
    
    def adjust_response_tone(self, sentiment_result):
        """Suggest response tone based on sentiment"""
        sentiment = sentiment_result['sentiment']
        emotions = sentiment_result['emotions']
        
        if sentiment == 'negative' or 'angry' in emotions or 'frustrated' in emotions:
            return 'empathetic'
        elif sentiment == 'positive' or 'happy' in emotions:
            return 'enthusiastic'
        elif 'sad' in emotions:
            return 'supportive'
        else:
            return 'neutral'
