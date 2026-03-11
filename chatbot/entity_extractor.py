# chatbot/entity_extractor.py
import re
from datetime import datetime
import spacy

class EntityExtractor:
    def __init__(self):
        # Try to load spaCy model, fallback to basic extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
        except:
            self.use_spacy = False
            print("SpaCy model not found. Using basic entity extraction.")
        
        # Custom entity patterns
        self.patterns = {
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'phone': r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}',
            'order_id': r'[A-Z]{2,3}[-]?\d{6,10}',
            'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'time': r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?',
            'money': r'\$\d+(?:\.\d{2})?|\d+\s*(?:dollars|USD)',
            'percentage': r'\d+(?:\.\d+)?%'
        }
    
    def extract_entities(self, text):
        """Extract all entities from text"""
        entities = {}
        
        # Extract custom pattern entities
        entities['custom'] = self._extract_custom_entities(text)
        
        # Extract spaCy entities if available
        if self.use_spacy:
            entities['spacy'] = self._extract_spacy_entities(text)
        
        # Flatten and combine entities
        combined_entities = self._combine_entities(entities)
        
        return combined_entities
    
    def _extract_custom_entities(self, text):
        """Extract entities using regex patterns"""
        extracted = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[entity_type] = matches
        
        return extracted
    
    def _extract_spacy_entities(self, text):
        """Extract entities using spaCy NER"""
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_.lower()
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(ent.text)
        
        return entities
    
    def _combine_entities(self, entities):
        """Combine all extracted entities"""
        combined = {}
        
        for source, entity_dict in entities.items():
            for entity_type, values in entity_dict.items():
                if entity_type not in combined:
                    combined[entity_type] = []
                combined[entity_type].extend(values)
        
        # Remove duplicates
        for entity_type in combined:
            combined[entity_type] = list(set(combined[entity_type]))
        
        return combined
    
    def extract_specific_entity(self, text, entity_type):
        """Extract a specific type of entity"""
        all_entities = self.extract_entities(text)
        return all_entities.get(entity_type, [])
    
    def get_entity_summary(self, text):
        """Get a summary of all entities found"""
        entities = self.extract_entities(text)
        summary = {
            'total_entities': sum(len(v) for v in entities.values()),
            'entity_types': list(entities.keys()),
            'entities': entities
        }
        return summary