import spacy
from typing import Optional, Dict, List, TypedDict
import logging
import re
import tldextract
from email.utils import parseaddr
from utils.filters.filter_repository import get_filter_repository

logger = logging.getLogger(__name__)

# Company Candidate Structure
class CompanyCandidate(TypedDict):
    name: str
    source: str  # 'span' | 'domain' | 'signature' | 'ner' | 'body'
    confidence: float  # 0.0 - 1.0
    type: str  # 'vendor' | 'client' | 'ats' | 'unknown'

# Scoring System (CSV-driven filtering makes this even more effective)
COMPANY_SOURCE_SCORES = {
    'span': 0.95,      # HTML span tags - highest confidence
    'domain': 0.90,    # Email domain - very reliable
    'signature': 0.80, # Email signature - reliable
    'body_intro': 0.65, # Body introduction - moderate
    'ner': 0.50       # NER extraction - lowest confidence
}

COMPANY_PENALTIES = {
    'ats_domain': -0.40,      # ATS platform detected
    'contains_client': -0.50, # Client language detected
    'generic_term': -0.30,    # Generic company terms
    'too_short': -0.20        # Company name too short
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
        
        # Penalty for too short
        if len(name) < 3:
            score += COMPANY_PENALTIES['too_short']
            self.logger.debug(f"Penalty: Too short ({name})")
        
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
            # CANDIDATE 1: HTML Span extraction (highest confidence)
            if html:
                vendor_info = self.extract_vendor_from_span(html)
                if vendor_info.get('company'):
                    candidate: CompanyCandidate = {
                        'name': vendor_info['company'],
                        'source': 'span',
                        'confidence': 0.0,
                        'type': 'vendor'
                    }
                    candidate['confidence'] = self._calculate_company_score(candidate, html)
                    candidates.append(candidate)
                    self.logger.debug(f"Candidate from span: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 2: Domain extraction (very reliable if not ATS)
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
                    self.logger.debug(f"Candidate from domain: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 3: Signature extraction
            sig_company = self.extract_company_from_signature(text)
            if sig_company:
                candidate: CompanyCandidate = {
                    'name': sig_company,
                    'source': 'signature',
                    'confidence': 0.0,
                    'type': 'unknown'
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.debug(f"Candidate from signature: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
            # CANDIDATE 4: NER extraction (lowest confidence)
            entities = self.extract_entities(text)
            if entities.get('company'):
                candidate: CompanyCandidate = {
                    'name': entities['company'],
                    'source': 'ner',
                    'confidence': 0.0,
                    'type': 'unknown'
                }
                candidate['confidence'] = self._calculate_company_score(candidate, text)
                candidates.append(candidate)
                self.logger.debug(f"Candidate from NER: {candidate['name']} (score: {candidate['confidence']:.2f})")
            
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
