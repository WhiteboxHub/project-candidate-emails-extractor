import spacy
from typing import Optional, Dict, List, TypedDict
import logging
import re
import tldextract
from email.utils import parseaddr
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)

# Company Candidate Structure
class CompanyCandidate(TypedDict):
    name: str
    source: str  # 'span' | 'domain' | 'signature' | 'ner' | 'body'
    confidence: float  # 0.0 - 1.0
    type: str  # 'vendor' | 'client' | 'ats' | 'unknown'

# Scoring System - PRIORITIZES CLIENT COMPANY (where job is) over VENDOR COMPANY (recruiting agency)
# Strategy: Extract the company where the POSITION is, not where the recruiter works
COMPANY_SOURCE_SCORES = {
    'client_explicit': 0.95,   # Explicit "client: XYZ" or "end client: ABC" - HIGHEST
    'span': 0.90,              # HTML span tags - very high confidence
    'body_client_pattern': 0.85, # "Position at Company" or "role with Company" patterns
    'signature': 0.75,         # Email signature - reliable but could be vendor
    'body_intro': 0.60,        # Body introduction - moderate (could be vendor intro)
    'ner': 0.50,               # NER extraction - moderate confidence
    'domain': 0.30             # Email domain - LOWEST (usually vendor, not client!)
}

COMPANY_PENALTIES = {
    'ats_domain': -0.40,       # ATS platform detected
    'contains_client': -0.50,  # Client language detected (paradoxically means it's NOT the client)
    'generic_term': -0.30,     # Generic company terms
    'too_short': -0.20,        # Company name too short
    'is_location': -0.60,      # Location detected (strong penalty to reject)
    'is_vendor_domain': -0.50  # NEW: Domain is from vendor email (not client company)
}

MIN_COMPANY_SCORE = 0.70  # Minimum score to accept candidate

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
        
        
        # Load filter repository
        self.filter_repo = get_filter_repository()
        
        # Load filter lists from CSV for company extraction
        self.job_title_keywords = self._load_job_title_keywords()
        self.company_suffixes = self._load_company_suffixes()
        self.ats_domains = self._load_ats_domains()
        self.client_keywords = self._load_client_keywords()
        self.generic_terms = self._load_generic_terms()
        self.vendor_indicators = self._load_vendor_indicators()
    
    def _load_job_title_keywords(self) -> set:
        """Load job title keywords from filter repository (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            if 'job_title_keywords' in keyword_lists:
                keywords = keyword_lists['job_title_keywords']
                # Convert to set and lowercase
                job_titles = {kw.lower().strip() for kw in keywords}
                self.logger.info(f"✓ Loaded {len(job_titles)} job title keywords from CSV")
                return job_titles
            else:
                self.logger.error("⚠ job_title_keywords not found in CSV - using empty set")
                return set()
                
        except Exception as e:
            self.logger.error(f"Failed to load job title keywords from CSV: {str(e)} - using empty set")
            return set()  # No hardcoded fallback - return empty set
    
    def _load_company_suffixes(self) -> dict:
        """Load company suffix mappings from filter repository (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            if 'company_suffix_mapping' in keyword_lists:
                # Parse suffix mappings from CSV (format: "old|new, old2|new2")
                mappings_str = keyword_lists['company_suffix_mapping']
                suffixes = {}
                
                for mapping in mappings_str:
                    if '|' in mapping:
                        old, new = mapping.split('|', 1)
                        suffixes[old.strip()] = new.strip()
                
                if suffixes:
                    self.logger.info(f"✓ Loaded {len(suffixes)} company suffix mappings from CSV")
                    return suffixes
                else:
                    self.logger.error("⚠ No valid suffix mappings found in CSV - using empty dict")
                    return {}
            else:
                self.logger.error("⚠ company_suffix_mapping not found in CSV - using empty dict")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to load company suffixes from CSV: {str(e)} - using empty dict")
            return {}  # No hardcoded fallback - return empty dict
    
    def _load_ats_domains(self) -> list:
        """Load ATS platform domains from CSV (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            # Check both old and new category names
            for category in ['blocked_ats_domain', 'ats_domains']:
                if category in keyword_lists:
                    domains = keyword_lists[category]
                    self.logger.info(f"✓ Loaded {len(domains)} ATS domains from CSV")
                    return domains
            
            self.logger.error("⚠ ATS domains not found in CSV - using empty list")
            return []
        except Exception as e:
            self.logger.error(f"Failed to load ATS domains from CSV: {str(e)} - using empty list")
            return []
    
    def _load_client_keywords(self) -> list:
        """Load client language keywords from CSV (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            if 'client_language_keywords' in keyword_lists:
                keywords = keyword_lists['client_language_keywords']
                self.logger.info(f"✓ Loaded {len(keywords)} client language keywords from CSV")
                return keywords
            else:
                self.logger.error("⚠ client_language_keywords not found in CSV - using empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load client keywords from CSV: {str(e)} - using empty list")
            return []
    
    def _load_generic_terms(self) -> list:
        """Load generic company terms from CSV (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            if 'generic_company_terms' in keyword_lists:
                terms = keyword_lists['generic_company_terms']
                self.logger.info(f"✓ Loaded {len(terms)} generic company terms from CSV")
                return terms
            else:
                self.logger.error("⚠ generic_company_terms not found in CSV - using empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load generic terms from CSV: {str(e)} - using empty list")
            return []
    
    def _load_vendor_indicators(self) -> list:
        """Load vendor indicator phrases from CSV (CSV only - no fallback)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            if 'vendor_indicators' in keyword_lists:
                indicators = keyword_lists['vendor_indicators']
                self.logger.info(f"✓ Loaded {len(indicators)} vendor indicators from CSV")
                return indicators
            else:
                self.logger.error("⚠ vendor_indicators not found in CSV - using empty list")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load vendor indicators from CSV: {str(e)} - using empty list")
            return []
    
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
                    # Filter out job titles and locations
                    company_candidate = ent.text.strip()
                    if not self._is_job_title(company_candidate) and not self._is_location(company_candidate):
                        entities['company'] = company_candidate
                    elif self._is_location(company_candidate):
                        self.logger.debug(f"Spacy NER: Rejected location classified as ORG: {company_candidate}")
                
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
    
    def _contains_client_language(self, text: str) -> bool:
        """Check if text contains client company indicators (CSV-driven, no hardcoded values)"""
        if not text or not self.client_keywords:
            return False
        
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.client_keywords)
    
    def _is_ats_domain(self, domain: str) -> bool:
        """Check if domain is an ATS platform (CSV-driven, no hardcoded values)"""
        if not domain or not self.ats_domains:
           return False
        
        domain_lower = domain.lower()
        return any(ats in domain_lower for ats in self.ats_domains)
    
    def _is_valid_company_candidate(self, company: str, context: str = "") -> bool:
        """
        Comprehensive validation to reject junk company data
        
        This method uses dynamic pattern detection to identify and reject:
        - Timestamps (AM PST, PM EST, etc.)
        - Email body text fragments
        - Tech stack descriptions
        - Sentence fragments
        - HTML entities
        - Single words without business context
        
        Args:
            company: Company name candidate
            context: Surrounding text for context analysis
            
        Returns:
            True if valid company, False if junk data
        """
        if not company or len(company) < 2:
            return False
        
        company_lower = company.lower()
        
        # 1. REJECT: Too long (likely a sentence fragment or email body text)
        if len(company) > 50:
            self.logger.debug(f"❌ Company too long (sentence fragment): {company}")
            return False
        
        # 2. REJECT: Timestamp patterns (AM PST, PM EST, 11:30 AM, etc.)
        timestamp_patterns = [
            r'^\d{1,2}:\d{2}\s*(AM|PM|am|pm)',  # 11:30 AM
            r'^(AM|PM)\s+(PST|EST|CST|MST|PDT|EDT|CDT|MDT)',  # AM PST
            r'^\d{1,2}\s*(AM|PM)',  # 11 AM
        ]
        if any(re.match(pattern, company) for pattern in timestamp_patterns):
            self.logger.debug(f"❌ Company is timestamp: {company}")
            return False
        
        # 3. REJECT: Tech stack descriptions
        tech_stack_indicators = [
            'tech stack:', 'python,', 'java,', 'aws (', 'docker,', 'kubernetes,',
            'langchain,', 'tensorflow,', 'pytorch,', 'react,', 'node.js,',
            'eks,', 'sagemaker,', 'lambda,', 'sql.'
        ]
        if any(indicator in company_lower for indicator in tech_stack_indicators):
            self.logger.debug(f"❌ Company is tech stack description: {company}")
            return False
        
        # 4. REJECT: Sentence fragments (contains sentence indicators)
        sentence_indicators = [
            '. ', '! ', '? ',  # Sentence endings
            ' and i ', ' to provide ', ' has reviewed ', ' wanted to ',
            ' we are ', ' at this time ', ' please ', ' thank you ',
            ' our team ', ' i am ', ' you are ', ' they are '
        ]
        if any(ind in company_lower for ind in sentence_indicators):
            self.logger.debug(f"❌ Company contains sentence fragment: {company}")
            return False
        
        # 5. REJECT: Starts with common sentence starters
        sentence_starters = [
            'our team', 'i wanted', 'thank you', 'please', 'we are',
            'at this time', 'i am', 'you are', 'they are', 'he is', 'she is',
            'it is', 'there are', 'there is', 'this is', 'that is'
        ]
        if any(company_lower.startswith(starter) for starter in sentence_starters):
            self.logger.debug(f"❌ Company starts with sentence: {company}")
            return False
        
        # 6. REJECT: HTML entities or encoding artifacts
        html_artifacts = ['&nbsp', '&amp', '&quot', '&lt', '&gt', '&#', '\u0026nbsp']
        if any(artifact in company for artifact in html_artifacts):
            self.logger.debug(f"❌ Company contains HTML entities: {company}")
            return False
        
        # 7. REJECT: Single common words (not business names)
        single_word_rejects = [
            'area', 'story', 'nbsp', 'quot', 'amp', 'team', 'group',
            'department', 'division', 'unit', 'office', 'branch'
        ]
        if company_lower in single_word_rejects:
            self.logger.debug(f"❌ Company is single common word: {company}")
            return False
        
        # 8. REJECT: Starts with lowercase (likely mid-sentence extraction)
        if company[0].islower():
            self.logger.debug(f"❌ Company starts with lowercase: {company}")
            return False
        
        # 9. REJECT: Contains excessive punctuation (likely junk)
        punctuation_count = sum(1 for c in company if c in '.,!?;:')
        if punctuation_count > 2:
            self.logger.debug(f"❌ Company has excessive punctuation: {company}")
            return False
        
        # 10. REJECT: Ends with incomplete sentence indicators
        incomplete_endings = [' and', ' or', ' the', ' a', ' an', ' to', ' for', ' with', ' in', ' on', ' at']
        if any(company_lower.endswith(ending) for ending in incomplete_endings):
            self.logger.debug(f"❌ Company ends with incomplete phrase: {company}")
            return False
        
        # 11. WARN: Very short without business suffix (might be valid, but low confidence)
        if len(company) < 4:
            business_suffixes = ['inc', 'llc', 'ltd', 'co']
            if not any(suffix in company_lower for suffix in business_suffixes):
                self.logger.debug(f"⚠️  Company very short without business suffix: {company}")
                # Don't reject, but this will get low score in _calculate_company_score
        
        # 12. CONTEXT CHECK: If company appears in a question context, likely junk
        if context:
            question_context = ['what ', 'why ', 'how ', 'when ', 'where ', 'who ', 'which ']
            # Check if company appears near a question word
            company_pos = context.lower().find(company_lower)
            if company_pos > 0:
                preceding_text = context[max(0, company_pos - 50):company_pos].lower()
                if any(q in preceding_text for q in question_context):
                    self.logger.debug(f"❌ Company appears in question context: {company}")
                    return False
        
        # Passed all validation checks
        return True
    
    def _calculate_company_score(self, candidate: CompanyCandidate, context: str = "") -> float:
        """Calculate confidence score for company candidate using scoring system"""
        # Start with base score from source
        score = COMPANY_SOURCE_SCORES.get(candidate['source'], 0.50)
        
        name = candidate['name']
        
        # Apply penalties
        if candidate['type'] == 'ats':
            score += COMPANY_PENALTIES['ats_domain']
            self.logger.debug(f"Penalty: ATS domain detected ({name})")
        
        if self._contains_client_language(context) or self._contains_client_language(name):
            score += COMPANY_PENALTIES['contains_client']
            candidate['type'] = 'client'
            self.logger.debug(f"Penalty: Client language detected ({name})")
        
        # Check for generic terms
        if self.generic_terms and any(term in name.lower() for term in self.generic_terms):
            score += COMPANY_PENALTIES['generic_term']
            self.logger.debug(f"Penalty: Generic term detected ({name})")
        
        # CRITICAL: Check if it's actually a location (strong penalty)
        if self._is_location(name):
            score += COMPANY_PENALTIES['is_location']
            self.logger.debug(f"Penalty: Location detected as company ({name}) - REJECTING")
        
        # Penalty for too short
        if len(name) < 3:
            score += COMPANY_PENALTIES['too_short']
            self.logger.debug(f"Penalty: Too short ({name})")
        
        # BONUS: Company has common business suffix (Inc, LLC, Corp, Ltd, etc.)
        company_suffixes = ['inc', 'llc', 'corp', 'ltd', 'limited', 'corporation', 'incorporated', 'co', 'company', 'group', 'solutions', 'services', 'technologies', 'tech', 'systems']
        if any(name.lower().endswith(suffix) or f' {suffix}' in name.lower() for suffix in company_suffixes):
            score += 0.10
            self.logger.debug(f"Bonus: Company suffix detected ({name})")
        
        # BONUS: Contains vendor indicators (staffing, recruiting, solutions, etc.)
        if self.vendor_indicators and any(indicator in name.lower() for indicator in self.vendor_indicators):
            score += 0.05
            candidate['type'] = 'vendor'
            self.logger.debug(f"Bonus: Vendor indicator detected ({name})")
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
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
        """Extract and format company name from email domain using tldextract
        
        Examples:
        - john@techcorp.com -> TechCorp
        - jane@cyber-coders.com -> Cyber Coders
        - jobs@lever.co -> None (ATS domain)
        - john@accenture.biz -> Accenture (root domain)
        """
        try:
            if not email or '@' not in email:
                return None
            
            full_domain = email.split('@')[1]
            
            # Check if it's a generic/personal domain using filter repository (CSV-driven)
            if self.filter_repo.check_email(email) == 'block':
                self.logger.debug(f"Blocked personal/generic domain: {full_domain}")
                return None
            
            # Use tldextract to get root domain (handles subdomains properly)
            ext = tldextract.extract(full_domain)
            company_name = ext.domain  # This is the root domain (e.g., 'accenture' from 'jobs.accenture.com')
            
            if not company_name:
                return None
            
            # Check if it's an ATS platform domain (CSV-driven)
            if self._is_ats_domain(full_domain):
                self.logger.debug(f"✗ Rejected ATS domain: {full_domain}")
                return None
            
            # Replace hyphens and underscores with spaces
            company_name = company_name.replace('-', ' ').replace('_', ' ')
            
            # Title case each word
            company_name = ' '.join(word.capitalize() for word in company_name.split())
            
            # Clean up with standard cleaning
            company_name = self._clean_company_name(company_name)
            
            if company_name:
                self.logger.debug(f"✓ Extracted company from domain: {company_name} (from {full_domain})")
            
            return company_name
            
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
    
    def _is_location(self, text: str) -> bool:
        """Check if text looks like a location (city, state, country) rather than a company name"""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[^\w\s]', '', text_lower)  # Remove punctuation
        
        # Common location indicators
        location_indicators = [
            # US States (abbreviations and full names)
            'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut',
            'delaware', 'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa',
            'kansas', 'kentucky', 'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan',
            'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire',
            'new jersey', 'new mexico', 'new york', 'north carolina', 'north dakota', 'ohio',
            'oklahoma', 'oregon', 'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
            'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington', 'west virginia',
            'wisconsin', 'wyoming',
            # State abbreviations
            'ca', 'ny', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi', 'nj', 'va', 'wa', 'az',
            'ma', 'tn', 'in', 'mo', 'md', 'wi', 'co', 'mn', 'sc', 'al', 'la', 'ky', 'or', 'ok',
            'ct', 'ia', 'ut', 'ar', 'nv', 'ms', 'ks', 'nm', 'ne', 'wv', 'id', 'hi', 'nh', 'me',
            'ri', 'mt', 'de', 'sd', 'nd', 'ak', 'dc', 'vt', 'wy',
            # Common location suffixes
            'city', 'town', 'county', 'state', 'province', 'region', 'area', 'district',
            # Common location patterns
            'united states', 'usa', 'us', 'uk', 'united kingdom', 'canada', 'australia',
            # Directional indicators (often part of location names)
            'north', 'south', 'east', 'west', 'northern', 'southern', 'eastern', 'western',
            'upper', 'lower', 'central', 'metro', 'greater'
        ]
        
        # Check if text contains location indicators (WITH WORD BOUNDARIES)
        # CRITICAL FIX: Use exact word matching for short indicators (like state codes 'ca', 'al')
        # otherwise 'Sibitalent' matches 'al' and gets rejected.
        text_words = set(text_clean.split())
        for indicator in location_indicators:
            # For short indicators (len <= 3), require exact match
            if len(indicator) <= 3:
                if indicator in text_words:
                    self.logger.debug(f"Rejected location as company: {text} (exact match '{indicator}')")
                    return True
            # For longer patterns ("united states", "california"), allow substring
            else:
                if indicator in text_clean:
                    self.logger.debug(f"Rejected location as company: {text} (contains '{indicator}')")
                    return True
        
        # Check if it's a common city name pattern (single word, capitalized, common city names)
        common_cities = [
            'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
            'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
            'san francisco', 'indianapolis', 'columbus', 'fort worth', 'charlotte',
            'seattle', 'denver', 'washington', 'boston', 'el paso', 'detroit', 'nashville',
            'portland', 'oklahoma city', 'las vegas', 'memphis', 'louisville', 'baltimore',
            'milwaukee', 'albuquerque', 'tucson', 'fresno', 'sacramento', 'kansas city',
            'mesa', 'atlanta', 'omaha', 'colorado springs', 'raleigh', 'virginia beach',
            'miami', 'oakland', 'minneapolis', 'tulsa', 'cleveland', 'wichita', 'arlington',
            'tampa', 'new orleans', 'honolulu', 'london', 'paris', 'tokyo', 'sydney',
            'toronto', 'vancouver', 'montreal', 'mumbai', 'delhi', 'bangalore', 'singapore'
        ]
        
        if text_clean in common_cities:
            self.logger.debug(f"Rejected known city as company: {text}")
            return True
        
        # Pattern: If text is just 1-2 words and looks like a location (all caps or title case, no numbers)
        # REMOVED AGGRESSIVE CHECK: This was rejecting valid single-word companies (e.g. "Google", "Stripe")
        # that don't have suffixes. We should rely on the explicit location lists above instead.
        
        return False
        
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
    
    def extract_company_from_body_intro(self, text: str) -> Optional[str]:
        """Extract company name from body introduction patterns
        
        Looks for patterns like:
        - "I'm from XYZ Company"
        - "I work at ABC Corp"
        - "I represent TechCorp"
        - "calling from XYZ Solutions"
        """
        try:
            # Common introduction patterns
            patterns = [
                # "I'm from/with/at Company"
                r"(?:I'?m|I am)\s+(?:from|with|at)\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
                # "I work for/at/with Company"
                r"(?:I|We)\s+work\s+(?:for|at|with)\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
                # "I represent Company"
                r"(?:I|We)\s+represent\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
                # "calling from Company"
                r"calling\s+from\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
                # "reaching out from Company"
                r"reaching\s+out\s+from\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
                # "Name - Title at Company"
                r"(?:^|\n)\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s*[-–—]\s*[A-Za-z\s]+\s+at\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|\n|$)",
                # "working with Company"
                r"working\s+with\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
                # "on behalf of Company"
                r"on\s+behalf\s+of\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+and\s|\s+in\s|\s+for\s|\s+to\s|$)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    potential_company = match.group(1).strip()
                    
                    # Clean up the match
                    potential_company = re.sub(r'\s+', ' ', potential_company)  # Normalize whitespace
                    potential_company = potential_company.strip('.,;: ')
                    
                    # Validate it looks like a company
                    if self._is_valid_company_name(potential_company):
                        cleaned = self._clean_company_name(potential_company)
                        self.logger.debug(f"✓ Extracted company from body intro: {cleaned}")
                        return cleaned
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting company from body intro: {str(e)}")
            return None
    
    def extract_client_company_explicit(self, text: str) -> Optional[str]:
        """Extract client company from explicit mentions - HIGHEST PRIORITY
        
        Looks for explicit client mentions like:
        - "Client: ABC Corp"
        - "End Client: XYZ Inc"
        - "Client Name: TechCorp"
        - "Our client, ABC Company"
        - "for our client ABC Corp"
        """
        try:
            # Explicit client patterns (case-insensitive)
            patterns = [
                # "Client: Company" or "End Client: Company"
                r"(?:end\s+)?client\s*:\s*([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+is\s|\s+has\s|\s+in\s|\n|$)",
                # "Client Name: Company"
                r"client\s+name\s*:\s*([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+is\s|\s+has\s|\s+in\s|\n|$)",
                # "Our client, Company" or "our client Company"
                r"our\s+client[,\s]+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+is\s|\s+has\s|\s+in\s|\n|$)",
                # "for our client Company"
                r"for\s+our\s+client\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+is\s|\s+has\s|\s+in\s|\n|$)",
                # "Client Company Name: XYZ"
                r"client\s+company\s+name\s*:\s*([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+is\s|\s+has\s|\s+in\s|\n|$)",
                # "working with client Company"
                r"working\s+with\s+(?:our\s+)?client\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+is\s|\s+has\s|\s+in\s|\n|$)",
                # "Position with [Company]" or "Position at [Company]" (in brackets/parentheses)
                r"position\s+(?:with|at)\s+\[([A-Z][a-zA-Z0-9\s&.,'-]+?)\]",
                r"position\s+(?:with|at)\s+\(([A-Z][a-zA-Z0-9\s&.,'-]+?)\)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    potential_company = match.group(1).strip()
                    
                    # Clean up the match
                    potential_company = re.sub(r'\s+', ' ', potential_company)
                    potential_company = potential_company.strip('.,;: ')
                    
                    # Validate it looks like a company
                    if self._is_valid_company_name(potential_company):
                        cleaned = self._clean_company_name(potential_company)
                        self.logger.info(f"✓✓✓ EXPLICIT CLIENT FOUND: {cleaned}")
                        return cleaned
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting explicit client company: {str(e)}")
            return None
    
    def extract_company_from_position_context(self, text: str) -> Optional[str]:
        """Extract client company from position context patterns
        
        Looks for patterns like:
        - "Java Developer at ABC Corp"
        - "Senior Engineer with XYZ Inc"
        - "role at TechCorp"
        - "position with ABC Company"
        """
        try:
            # Position context patterns
            patterns = [
                # "Position/Role/Job at Company"
                r"(?:position|role|job|opportunity)\s+(?:at|with)\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+in\s|\s+for\s|\s+located\s|\n|$)",
                # "Job Title at Company" (e.g., "Java Developer at ABC Corp")
                r"(?:developer|engineer|analyst|manager|architect|consultant|specialist|lead|senior|junior)\s+at\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+in\s|\s+for\s|\s+located\s|\n|$)",
                # "Job Title with Company"
                r"(?:developer|engineer|analyst|manager|architect|consultant|specialist|lead|senior|junior)\s+with\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+in\s|\s+for\s|\s+located\s|\n|$)",
                # "opening at Company"
                r"opening\s+at\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+in\s|\s+for\s|\s+located\s|\n|$)",
                # "vacancy at Company"
                r"vacancy\s+at\s+([A-Z][a-zA-Z0-9\s&.,'-]+?)(?:\.|,|;|\s+in\s|\s+for\s|\s+located\s|\n|$)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
                if match:
                    potential_company = match.group(1).strip()
                    
                    # Clean up the match
                    potential_company = re.sub(r'\s+', ' ', potential_company)
                    potential_company = potential_company.strip('.,;: ')
                    
                    # Validate it looks like a company
                    if self._is_valid_company_name(potential_company):
                        cleaned = self._clean_company_name(potential_company)
                        self.logger.debug(f"✓ Extracted company from position context: {cleaned}")
                        return cleaned
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting company from position context: {str(e)}")
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
        
        # Must not be a location
        if self._is_location(text):
            return False
        
        # Must have at least some letters
        if not any(c.isalpha() for c in text):
            return False
        
        # Not too long (no company name should be > 100 chars)
        if len(text) > 100:
            return False
        
        return True
    
    def extract_company_with_scoring(self, text: str, email: str = None, html: str = None) -> Optional[str]:
        """
        Extract company name using scoring system to pick best candidate
        
        This is the MAIN entry point for noise-free company extraction.
        Collects candidates from all sources, scores them, returns the best one.
        
        Args:
            text: Email body text (cleaned)
            email: Sender email address  
            html: Raw HTML body (for span extraction)
            
        Returns:
            Best company name or None
        """
        candidates: List[CompanyCandidate] = []
        
        try:
            # CANDIDATE 1: EXPLICIT CLIENT MENTIONS (HIGHEST PRIORITY - 0.95)
            # "Client: ABC Corp", "End Client: XYZ", "Our client, TechCorp"
            explicit_client = self.extract_client_company_explicit(text)
            if explicit_client and self._is_valid_company_candidate(explicit_client, text):
                candidate: CompanyCandidate = {
                    'name': explicit_client,
                    'source': 'client_explicit',
                    'confidence': 0.0,
                    'type': 'client'
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.info(f"🎯 Candidate from EXPLICIT CLIENT: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 2: HTML Span extraction (0.90)
            if html:
                vendor_info = self.extract_vendor_from_span(html)
                if vendor_info.get('company') and self._is_valid_company_candidate(vendor_info['company'], html):
                    candidate: CompanyCandidate = {
                        'name': vendor_info['company'],
                        'source': 'span',
                        'confidence': 0.0,
                        'type': 'unknown'  # Could be client or vendor
                    }
                    candidate['confidence'] = self._calculate_company_score(candidate, html)
                    candidates.append(candidate)
                    self.logger.debug(f"Candidate from span: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 3: Position Context Patterns (0.85)
            # "Java Developer at ABC Corp", "role with XYZ Inc"
            position_company = self.extract_company_from_position_context(text)
            if position_company and self._is_valid_company_candidate(position_company, text):
                candidate: CompanyCandidate = {
                    'name': position_company,
                    'source': 'body_client_pattern',
                    'confidence': 0.0,
                    'type': 'client'  # Position context usually means client
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.debug(f"Candidate from position context: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 4: Signature extraction (0.75)
            sig_company = self.extract_company_from_signature(text)
            if sig_company and self._is_valid_company_candidate(sig_company, text):
                candidate: CompanyCandidate = {
                    'name': sig_company,
                    'source': 'signature',
                    'confidence': 0.0,
                    'type': 'unknown'  # Could be vendor or client
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.debug(f"Candidate from signature: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 5: Body introduction extraction (0.60)
            # "I'm from XYZ" - usually vendor introducing themselves
            body_intro_company = self.extract_company_from_body_intro(text)
            if body_intro_company and self._is_valid_company_candidate(body_intro_company, text):
                candidate: CompanyCandidate = {
                    'name': body_intro_company,
                    'source': 'body_intro',
                    'confidence': 0.0,
                    'type': 'vendor'  # Intro usually means vendor
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.debug(f"Candidate from body intro: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 6: NER extraction (0.50)
            entities = self.extract_entities(text)
            if entities.get('company') and self._is_valid_company_candidate(entities['company'], text):
                candidate: CompanyCandidate = {
                    'name': entities['company'],
                    'source': 'ner',
                    'confidence': 0.0,
                    'type': 'unknown'
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.debug(f"Candidate from NER: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 7: Domain extraction (0.30 - LOWEST PRIORITY!)
            # Email domain usually extracts VENDOR company, not CLIENT company
            # Only use as last resort fallback
            if email:
                domain_company = self.extract_company_from_domain(email)
                if domain_company:
                    full_domain = email.split('@')[1]
                    candidate_type = 'ats' if self._is_ats_domain(full_domain) else 'vendor'
                    
                    candidate: CompanyCandidate = {
                        'name': domain_company,
                        'source': 'domain',
                        'confidence': 0.0,
                        'type': candidate_type
                    }
                    candidate['confidence'] = self._calculate_company_score(candidate, text)
                    candidates.append(candidate)
                    self.logger.debug(f"Candidate from domain (VENDOR): {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # Filter candidates by minimum score
            valid_candidates = [c for c in candidates if c['confidence'] >= MIN_COMPANY_SCORE]
            
            if not valid_candidates:
                self.logger.debug(f"❌ No candidates met minimum score of {MIN_COMPANY_SCORE}")
                return None
            
            # Sort by confidence (highest first)
            valid_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return best candidate (prefer vendor over client if scores are close)
            best = valid_candidates[0]
            
            # Check if there's a vendor candidate close in score to a client
            for candidate in valid_candidates[1:]:
                if candidate['type'] == 'vendor' and best['type'] == 'client':
                    # If vendor candidate is within 0.15 of client, prefer vendor
                    if candidate['confidence'] >= best['confidence'] - 0.15:
                        best = candidate
                        self.logger.info(f"✓ Preferred vendor over client: {best['name']}")
                        break
            
            self.logger.info(f"✅ Best company: {best['name']} (source: {best['source']}, score: {best['confidence']:.2f}, type: {best['type']})")
            return best['name']
            
        except Exception as e:
            self.logger.error(f"Error in company extraction with scoring: {str(e)}")
            return None
    
    def _clean_company_name(self, company: str) -> str:
        """Clean and standardize company name"""
        if not company:
            return company
        
        # Remove extra whitespace
        company = ' '.join(company.split())
        
        # Remove trailing punctuation (but keep . for Inc., LLC., etc.)
        company = company.rstrip(',;: ')
        
        # Standardize common suffixes (loaded from CSV)
        company_lower = company.lower()
        for old, new in self.company_suffixes.items():
            if company_lower.endswith(old):
                company = company[:-len(old)] + new
                break
        
        return company
