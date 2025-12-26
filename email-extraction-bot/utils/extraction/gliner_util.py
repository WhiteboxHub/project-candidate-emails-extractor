from gliner import GLiNER
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class GLiNERExtractor:
    """
    Extract entities using GLiNER - zero-shot NER model
    More flexible and accurate than traditional BERT-based models
    """
    
    def __init__(self, config: dict):
        self.logger = logging.getLogger(__name__)
        
        # Load config
        gliner_config = config.get('gliner', {})
        model_name = gliner_config.get('model', 'urchade/gliner_base')
        self.threshold = gliner_config.get('threshold', 0.6)  # Higher threshold for better accuracy
        self.entity_labels = gliner_config.get('entity_labels', [
            'person name', 'full name', 'recruiter name',
            'company name', 'organization', 'employer',
            'city', 'location', 'address',
            'job title', 'position', 'role',
            'email address', 'phone number'
        ])
        
        try:
            self.model = GLiNER.from_pretrained(model_name)
            self.logger.info(f"GLiNER model loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load GLiNER: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract contact entities from text with smart pre-processing
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with keys: name, company, location, job_title
        """
        try:
            if not text or len(text.strip()) < 20:
                return {'name': None, 'company': None, 'location': None, 'job_title': None}
            
            # Extract signature section (most reliable for contact info)
            signature_text = self._extract_signature_section(text)
            
            # Use signature if available, otherwise use full text
            extraction_text = signature_text if signature_text else text[:2000]
            
            # Extract entities
            entities_raw = self.model.predict_entities(
                extraction_text, 
                self.entity_labels,
                threshold=self.threshold,
                flat_ner=True  # Better for overlapping entities
            )
            
            # Parse results
            return self._parse_entities(entities_raw)
            
        except Exception as e:
            self.logger.error(f"GLiNER extraction error: {str(e)}")
            return {
                'name': None, 
                'company': None, 
                'location': None, 
                'job_title': None
            }
    
    def _extract_signature_section(self, text: str) -> str:
        """Extract signature section from email (last 500 chars usually)"""
        try:
            # Look for common signature indicators
            signature_markers = [
                'Best regards', 'Best Regards', 'Regards', 'Thanks', 
                'Thank you', 'Sincerely', 'Warm regards', 'Cheers'
            ]
            
            text_lower = text.lower()
            for marker in signature_markers:
                pos = text_lower.rfind(marker.lower())
                if pos != -1:
                    # Get text from marker onwards (up to 500 chars)
                    return text[pos:pos+500]
            
            # Fallback: last 500 chars
            return text[-500:] if len(text) > 500 else text
            
        except:
            return text[:2000]
    
    def _parse_entities(self, entities_raw: List[Dict]) -> Dict[str, str]:
        """Parse GLiNER output to standardized format with robust validation"""
        entities = {
            'name': None,
            'company': None,
            'location': None,
            'job_title': None
        }
        
        # Blacklist of literal field labels and invalid patterns
        name_blacklist = {
            'full name', 'name', 'first name', 'last name', 'middle name',
            'your name', 'applicant name', 'candidate name', 'recruiter name',
            'contact name', 'person name', 'fullname', 'firstname', 'lastname'
        }
        
        company_blacklist = {
            'company', 'organization', 'firm', 'team', 'employer',
            'company name', 'organization name', 'employer name',
            'to:', 'from:', 'subject:', 're:', 'fwd:'
        }
        
        # Group by label type with scores
        candidates = {
            'name': [],
            'company': [],
            'location': [],
            'job_title': []
        }
        
        for entity in entities_raw:
            label = entity['label'].lower()
            text = entity['text'].strip()
            score = entity.get('score', 0)
            
            # Skip low confidence
            if score < self.threshold:
                continue
            
            # Skip obviously bad extractions
            if len(text) < 2 or len(text) > 100:
                continue
            
            text_lower = text.lower().strip()
            
            # Categorize entities with strict validation
            if 'person' in label or 'full name' in label or 'recruiter' in label:
                # Reject literal labels
                if text_lower in name_blacklist:
                    continue
                
                # Reject if it contains common field label patterns
                if any(pattern in text_lower for pattern in [':', '|', 'along with', 'as per', 'please provide']):
                    continue
                
                words = text.split()
                # Valid names: 2-4 words, no numbers, proper capitalization
                if 2 <= len(words) <= 4 and not any(char.isdigit() for char in text):
                    # Check if it looks like a real name (has proper capitalization)
                    if any(word[0].isupper() for word in words if word):
                        candidates['name'].append((text, score))
            
            elif 'company' in label or 'organization' in label or 'employer' in label:
                # Reject literal labels
                if text_lower in company_blacklist:
                    continue
                
                # Reject if it looks like a sentence or instruction
                if any(pattern in text_lower for pattern in [
                    'along with', 'as per', 'please', 'provide', 'fill', 'enter',
                    'to:', 'from:', 'subject:', 're:', 'fwd:', 'waiting for',
                    'details for', 'application form', 'confirmation'
                ]):
                    continue
                
                # Reject if it's too long (likely a sentence)
                if len(text.split()) > 8:
                    continue
                
                # Skip generic company terms
                if text_lower not in ['company', 'organization', 'firm', 'team']:
                    candidates['company'].append((text, score))
            
            elif 'location' in label or 'city' in label or 'address' in label:
                # Valid locations: no emails, no phone numbers, not too long
                if '@' not in text and len(text.split()) <= 5:
                    # Reject if it contains common non-location patterns
                    if not any(pattern in text_lower for pattern in [
                        'along with', 'as per', 'please', 'provide', 'to:', 'from:'
                    ]):
                        candidates['location'].append((text, score))
            
            elif 'job title' in label or 'position' in label or 'role' in label:
                # Reject if it's too long or contains invalid patterns
                if len(text.split()) <= 6 and not any(pattern in text_lower for pattern in [
                    'along with', 'as per', 'please', 'provide', 'to:', 'from:'
                ]):
                    candidates['job_title'].append((text, score))
        
        # Select best candidate for each field (highest score)
        for field, items in candidates.items():
            if items:
                # Sort by score descending
                items.sort(key=lambda x: x[1], reverse=True)
                entities[field] = items[0][0]  # Take highest scored
        
        return entities
