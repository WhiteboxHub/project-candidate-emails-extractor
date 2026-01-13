import spacy
from typing import Optional, Dict
import logging
import re
from email.utils import parseaddr

logger = logging.getLogger(__name__)

class SpacyNERExtractor:
    """Extract entities using Spacy NER"""
    
    def __init__(self, model: str = 'en_core_web_sm'):
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load(model)
            self.logger.info(f"Loaded Spacy model: {model}")
        except OSError:
            self.logger.error(f"Spacy model '{model}' not found. Run: python -m spacy download {model}")
            raise
        
        # Job titles to filter out (should not be extracted as company names)
        self.job_title_keywords = {
            # Recruiting roles
            'recruiter', 'talent acquisition', 'talent specialist', 'headhunter',
            'staffing', 'sourcer', 'recruitment', 'hiring',
            # Management titles
            'manager', 'director', 'lead', 'head', 'chief', 'president', 'vp',
            'vice president', 'senior', 'junior', 'principal', 'executive',
            # HR roles
            'hr', 'human resources', 'people operations', 'people ops',
            # Technical roles
            'engineer', 'developer', 'architect', 'designer', 'analyst',
            'consultant', 'specialist', 'coordinator', 'administrator',
            # Executive titles
            'ceo', 'cto', 'cfo', 'coo', 'cmo', 'founder', 'co-founder', 'partner',
            # Generic roles
            'associate', 'representative', 'advisor', 'agent', 'officer'
        }
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract named entities from text
        
        Returns:
            Dictionary with keys: name, company, location
        """
        try:
            doc = self.nlp(text)
            
            entities = {
                'name': None,
                'company': None,
                'location': None
            }
            
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and not entities['name']:
                    # Filter out single-word names (likely false positives)
                    if len(ent.text.split()) >= 2 and len(ent.text.split()) <= 3:
                        entities['name'] = ent.text.strip()
                
                elif ent.label_ == 'ORG' and not entities['company']:
                    # Filter out job titles
                    company_candidate = ent.text.strip()
                    if not self._is_job_title(company_candidate):
                        entities['company'] = company_candidate
                
                elif ent.label_ in ['GPE', 'LOC'] and not entities['location']:
                    entities['location'] = ent.text.strip()
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in Spacy NER extraction: {str(e)}")
            return {'name': None, 'company': None, 'location': None}
    
    def extract_name_from_signature(self, text: str) -> Optional[str]:
        """Extract name from email signature patterns with better patterns"""
        try:
            # Enhanced signature patterns
            patterns = [
                # After greeting with newline
                r'(?:Thanks|Regards|Best|Sincerely|Warm regards|Kind regards|Cheers),?\s*[\r\n]+\s*([A-Z][a-z]+(?:[\s-][A-Z][a-z]+){1,2})\s*[\r\n]',
                # Name followed by title/company
                r'([A-Z][a-z]+(?:[\s-][A-Z][a-z]+){1,2})\s*[\r\n]+(?:Senior|Lead|Director|Manager|Recruiter|VP|President)',
                # Name followed by phone or email on next line
                r'([A-Z][a-z]+(?:[\s-][A-Z][a-z]+){1,2})\s*[\r\n]+(?:Phone|Mobile|Email|Tel):',
                # Simple pattern
                r'(?:Thanks|Regards|Best|Sincerely),?\s*[\r\n]+\s*([A-Z][a-z]+(?:[\s][A-Z][a-z]+){1,2})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    # Validate
                    words = name.split()
                    if 2 <= len(words) <= 3 and not any(c.isdigit() for c in name):
                        return name
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting name from signature: {str(e)}")
            return None
    
    def extract_vendor_from_span(self, text: str) -> Dict[str, Optional[str]]:
        """Extract vendor name and company from HTML span tags or similar patterns
        
        Pattern examples:
        - <span>Name - Company</span>
        - <span>Name | Company</span>
        - <span>Name, Company</span>
        - <span>Name (Company)</span>
        - Plain text: Name - Company
        
        Returns:
            Dictionary with keys: name, company
        """
        try:
            # Multiple patterns to try (ordered by reliability)
            patterns = [
                # Pattern 1: HTML tags with Name - Company (hyphen separator)
                r'<(?:span|div|p|td|th|b|strong)[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[-–—]\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*</(?:span|div|p|td|th|b|strong)>',
                # Pattern 2: HTML tags with Name | Company (pipe separator)
                r'<(?:span|div|p|td|th|b|strong)[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\|\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*</(?:span|div|p|td|th|b|strong)>',
                # Pattern 3: HTML tags with Name, Company (comma separator)
                r'<(?:span|div|p|td|th|b|strong)[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*,\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*</(?:span|div|p|td|th|b|strong)>',
                # Pattern 4: HTML tags with Name (Company) (parentheses)
                r'<(?:span|div|p|td|th|b|strong)[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\(\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*\)\s*</(?:span|div|p|td|th|b|strong)>',
                # Pattern 5: Plain text with Name - Company (for text emails)
                r'(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[-–—]\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*(?:$|\n)',
                # Pattern 6: Plain text with Name | Company
                r'(?:^|\n)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*\|\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*(?:$|\n)',
                # Pattern 7: Name at Company format
                r'<(?:span|div|p)[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+at\s+([A-Z][a-zA-Z0-9\s&.,]+?)\s*</(?:span|div|p)>',
                # Pattern 8: Signature-style Name\nCompany (newline separated in HTML)
                r'<(?:span|div|p|b|strong)[^>]*>\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*</(?:span|div|p|b|strong)>\s*(?:<br\s*/?>|\n)\s*<(?:span|div|p)[^>]*>\s*([A-Z][a-zA-Z0-9\s&.,]+?)\s*</(?:span|div|p)>',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    name = match.group(1).strip()
                    company = match.group(2).strip()
                    
                    # Validate name (2-4 words, no digits, no special chars except space and hyphen)
                    name_words = name.split()
                    if 2 <= len(name_words) <= 4 and not any(c.isdigit() for c in name):
                        # Clean company name
                        # Remove HTML tags, extra whitespace, trailing punctuation
                        company = re.sub(r'<[^>]+>', '', company)  # Remove any HTML tags
                        company = re.sub(r'\s+', ' ', company)      # Normalize whitespace
                        company = company.strip('.,;: ')
                        
                        # Validate company (not empty, not too long, has letters)
                        if company and 1 < len(company) < 100 and any(c.isalpha() for c in company):
                            self.logger.info(f"✓ Extracted vendor from pattern: {name} - {company}")
                            return {'name': name, 'company': company}
            
            return {'name': None, 'company': None}
        except Exception as e:
            self.logger.error(f"Error extracting vendor from span: {str(e)}")
            return {'name': None, 'company': None}
    
    def extract_name_from_header(self, email_message) -> Optional[str]:
        """Extract name from email From header"""
        try:
            from_header = email_message.get('From', '')
            if not from_header:
                return None
            
            # Parse email header
            name, email_addr = parseaddr(from_header)
            
            # Clean up the name
            if name:
                # Remove quotes
                name = name.strip('"\' ')
                
                # Skip if it's just an email address
                if '@' in name:
                    return None
                
                # Skip if too short or too long
                words = name.split()
                if len(words) < 2 or len(words) > 4:
                    return None
                
                # Skip if has numbers (likely username)
                if any(char.isdigit() for char in name):
                    return None
                
                return name.strip()
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting name from header: {str(e)}")
            return None
    
    def extract_company_from_domain(self, email: str) -> Optional[str]:
        """Extract and format company name from email domain
        
        Examples:
        - john@techcorp.com -> TechCorp
        - jane@cyber-coders.com -> Cyber Coders
        - bob@acme-inc.com -> Acme Inc.
        """
        try:
            if not email or '@' not in email:
                return None
            
            # Blacklist of generic domains
            generic_domains = {
                'gmail', 'yahoo', 'hotmail', 'outlook', 'protonmail',
                'icloud', 'aol', 'mail', 'live', 'msn'
            }
            
            domain = email.split('@')[1]
            company_name = domain.split('.')[0]
            
            if company_name.lower() in generic_domains:
                return None
            
            # Replace hyphens and underscores with spaces
            company_name = company_name.replace('-', ' ').replace('_', ' ')
            
            # Title case each word
            company_name = ' '.join(word.capitalize() for word in company_name.split())
            
            # Clean up with standard cleaning
            company_name = self._clean_company_name(company_name)
            
        except Exception as e:
            self.logger.error(f"Error extracting company from domain: {str(e)}")
            return None
    
    def _is_job_title(self, text: str) -> bool:
        """Check if text is likely a job title rather than a company name"""
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check if any job title keyword appears in the text
        for keyword in self.job_title_keywords:
            if keyword in text_lower:
                self.logger.debug(f"Rejected job title as company: {text}")
                return True
        
        return False
    
    def extract_company_from_signature(self, text: str) -> Optional[str]:
        """Extract company name from email signature with pattern matching
        
        Looks for patterns like:
        John Smith
        Senior Recruiter
        TechCorp Inc.
        """
        try:
            # Look for company-like text after job title in signature
            lines = text.split('\n')
            
            for i, line in enumerate(lines):
                line_clean = line.strip()
                
                # If this line looks like a job title, next line might be company
                if self._is_job_title(line_clean) and i + 1 < len(lines):
                    potential_company = lines[i + 1].strip()
                    
                    # Validate it looks like a company
                    if self._is_valid_company_name(potential_company):
                        return self._clean_company_name(potential_company)
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting company from signature: {str(e)}")
            return None
    
    def _is_valid_company_name(self, text: str) -> bool:
        """Validate if text looks like a company name"""
        if not text or len(text) < 2:
            return False
        
        # Must start with capital letter or number
        if not (text[0].isupper() or text[0].isdigit()):
            return False
        
        # Must not be a job title
        if self._is_job_title(text):
            return False
        
        # Must have at least some letters
        if not any(c.isalpha() for c in text):
            return False
        
        # Not too long (no company name should be > 100 chars)
        if len(text) > 100:
            return False
        
        return True
    
    def _clean_company_name(self, company: str) -> str:
        """Clean and standardize company name"""
        if not company:
            return company
        
        # Remove extra whitespace
        company = ' '.join(company.split())
        
        # Remove trailing punctuation (but keep . for Inc., LLC., etc.)
        company = company.rstrip(',;: ')
        
        # Standardize common suffixes
        suffixes = {
            ' inc': ' Inc.',
            ' llc': ' LLC',
            ' corp': ' Corp.',
            ' ltd': ' Ltd.',
            ' co': ' Co.'
        }
        
        company_lower = company.lower()
        for old, new in suffixes.items():
            if company_lower.endswith(old):
                company = company[:-len(old)] + new
                break
        
        return company
