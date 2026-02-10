import re
import logging
from typing import Optional, Dict, List
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)


class LocationExtractor:
    """Extract locations and zip codes from email text"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize these before loading from CSV
        self.us_states = set()
        self.state_name_to_abbr = {}
        self.street_name_indicators = set()
        
        # Common phrases that are NOT city names (semantic validation)
        self.common_phrases = {
            'thank you', 'kind regards', 'best regards', 'sincerely',
            'regards', 'thanks', 'cheers', 'yours', 'respectfully',
            'cordially', 'warmly', 'looking forward'
        }
        
        # Common verbs/adjectives that are NOT city names
        self.common_verbs_adjectives = {
            'growing', 'managing', 'leading', 'developing', 'building',
            'creating', 'designing', 'testing', 'working', 'including',
            'ensuring', 'providing', 'supporting', 'maintaining'
        }
        
        # Technology terms that are NOT city names
        self.tech_terms = {
            'sql', 'api', 'aws', 'gcp', 'azure', 'cloud', 'java',
            'python', 'react', 'node', 'docker', 'kubernetes'
        }
        
        # Conjunctions/prepositions that shouldn't start city names
        self.invalid_prefixes = {'or', 'and', 'for', 'with', 'from', 'to'}
        
        # Load filter repository for CSV-driven configuration
        self.filter_repo = get_filter_repository()
        self._load_location_filters()
        
        # US ZIP code patterns
        self.us_zip_pattern = re.compile(
            r'\b(\d{5}(?:-\d{4})?)\b'  # 12345 or 12345-6789
        )
        
        # Canada postal code pattern
        self.canada_zip_pattern = re.compile(
            r'\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b',  # A1A 1A1 or A1A1A1
            re.IGNORECASE
        )
        
        # UK postal code pattern
        self.uk_zip_pattern = re.compile(
            r'\b([A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2})\b',  # SW1A 1AA, EC1A 1BB, etc.
            re.IGNORECASE
        )
        
        # Location patterns (City, State ZIP) - STRICT with word boundaries
        # Pattern explanation:
        # \b - Word boundary (prevents "Or Dallas", "Lo Alto")
        # [A-Z][a-z]+ - Proper capitalization (prevents "SQL", "ID", "THANK")
        # (?:\s+[A-Z][a-z]+)* - Multi-word cities (Palo Alto, Santa Clara)
        # {3,30} - Length validation (prevents single letters and very long phrases)
        self.location_patterns = [
            # "City, ST 12345" - STRICT State case [A-Z]{2}
            r'\b([A-Z][a-z]+(?:\s+[A-Za-z][a-z]+){0,3}),\s*([A-Z]{2})\b(?:\s*(\d{5}(?:-\d{4})?))?',
            
            # "Location: City, ST" - STRICT State case [A-Z]{2}
            r'(?:Location|City|Based in|Located in):\s*([A-Z][a-z]+(?:\s+[A-Za-z][a-z]+){0,3}),\s*([A-Z]{2})\b(?:\s*(\d{5}(?:-\d{4})?))?',
        ]
    
    def _load_location_filters(self):
        """Load location validation filters from CSV"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            # Load location false positives (junk words)
            if 'location_false_positives' in keyword_lists:
                self.location_false_positives = set(
                    kw.lower().strip() for kw in keyword_lists['location_false_positives']
                )
                self.logger.info(f"✓ Loaded {len(self.location_false_positives)} location false positives from CSV")
            else:
                self.location_false_positives = set()
                self.logger.warning("⚠ location_false_positives not found in CSV")
            
            # Load US major cities for validation
            if 'us_major_cities' in keyword_lists:
                self.us_major_cities = set(
                    kw.lower().strip() for kw in keyword_lists['us_major_cities']
                )
                self.logger.info(f"✓ Loaded {len(self.us_major_cities)} US cities from CSV")
            else:
                self.us_major_cities = set()
                self.logger.warning("⚠ us_major_cities not found in CSV")
            
            # Load location junk patterns (regex)
            if 'location_junk_patterns' in keyword_lists:
                self.location_junk_patterns = [
                    re.compile(pattern.strip(), re.IGNORECASE)
                    for pattern in keyword_lists['location_junk_patterns']
                ]
                self.logger.info(f"✓ Loaded {len(self.location_junk_patterns)} location junk patterns from CSV")
            else:
                self.location_junk_patterns = []
                self.logger.warning("⚠ location_junk_patterns not found in CSV")
            
            # Load US state abbreviations
            if 'us_state_abbreviations' in keyword_lists:
                self.us_states = set(
                    kw.upper().strip() for kw in keyword_lists['us_state_abbreviations']
                )
                self.logger.info(f"✓ Loaded {len(self.us_states)} US state abbreviations from CSV")
            else:
                self.us_states = set()
                self.logger.warning("⚠ us_state_abbreviations not found in CSV")
            
            # Load state name to abbreviation mappings (format: "name|abbr")
            if 'us_state_name_mappings' in keyword_lists:
                self.state_name_to_abbr = {}
                for mapping in keyword_lists['us_state_name_mappings']:
                    if '|' in mapping:
                        name, abbr = mapping.split('|', 1)
                        self.state_name_to_abbr[name.lower().strip()] = abbr.upper().strip()
                self.logger.info(f"✓ Loaded {len(self.state_name_to_abbr)} state name mappings from CSV")
            else:
                self.state_name_to_abbr = {}
                self.logger.warning("⚠ us_state_name_mappings not found in CSV")
            
            # Load street name indicators
            if 'location_name_indicators' in keyword_lists:
                self.street_name_indicators = set(
                    kw.lower().strip() for kw in keyword_lists['location_name_indicators']
                )
                self.logger.info(f"✓ Loaded {len(self.street_name_indicators)} street name indicators from CSV")
            else:
                self.street_name_indicators = set()
                self.logger.warning("⚠ location_name_indicators not found in CSV")
                
        except Exception as e:
            self.logger.error(f"Failed to load location filters from CSV: {str(e)}")
            self.location_false_positives = set()
            self.us_major_cities = set()
            self.location_junk_patterns = []
            self.us_states = set()
            self.state_name_to_abbr = {}
            self.street_name_indicators = set()
    
    def extract_zip_code(self, text: str) -> Optional[str]:
        """
        Extract ZIP/postal code from text
        """
        try:
            if not text:
                return None
            
            # Try US ZIP first (most common)
            match = self.us_zip_pattern.search(text)
            if match:
                zip_code = match.group(1)
                # Validate it's not a phone number or other number
                if self._is_valid_us_zip(zip_code):
                    self.logger.debug(f"✓ Extracted US ZIP: {zip_code}")
                    return zip_code
            
            # Try Canada postal code
            match = self.canada_zip_pattern.search(text)
            if match:
                zip_code = match.group(1).upper()
                self.logger.debug(f"✓ Extracted Canada postal code: {zip_code}")
                return zip_code
            
            # Try UK postal code
            match = self.uk_zip_pattern.search(text)
            if match:
                zip_code = match.group(1).upper()
                self.logger.debug(f"✓ Extracted UK postal code: {zip_code}")
                return zip_code
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting zip code: {str(e)}")
            return None
    
    def extract_location_with_zip(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract location with city, state, and zip code
        """
        result = {
            'location': None,
            'city': None,
            'state': None,
            'zip_code': None
        }
        
        try:
            if not text:
                return result
            
            # Try each location pattern
            for pattern in self.location_patterns:
                # Removed re.IGNORECASE to prevent matching "or" as "OR" (Oregon)
                # and to ensure proper capitalization of cities/states.
                matches = re.finditer(pattern, text, re.MULTILINE)
                
                for match in matches:
                    city = match.group(1).strip() if match.lastindex >= 1 else None
                    state = match.group(2).strip() if match.lastindex >= 2 else None
                    zip_code = match.group(3).strip() if match.lastindex >= 3 else None
                    
                    # Validate and normalize
                    if city and state:
                        # Normalize state to abbreviation
                        state_normalized = self._normalize_state(state)
                        
                        if state_normalized:
                            # Clean city name
                            city_clean = self._clean_city_name(city)
                            
                            if city_clean:
                                # Build location string
                                location = f"{city_clean}, {state_normalized}"
                                
                                # Validate zip if present
                                if zip_code and not self._is_valid_us_zip(zip_code):
                                    zip_code = None
                                
                                result['location'] = location
                                result['city'] = city_clean
                                result['state'] = state_normalized
                                result['zip_code'] = zip_code
                                
                                self.logger.debug(f"✓ Extracted location: {location}")
                                return result
            
            # If no structured location found, try to extract zip code separately from the whole text
            if not result['zip_code']:
                result['zip_code'] = self.extract_zip_code(text)
            
            # If we have a city/state but no zip code, try to find a zip code anywhere in the text 
            # (recruiter emails often have zip codes in signatures or separate lines)
            if (result['location'] or result['city']) and not result['zip_code']:
                result['zip_code'] = self.extract_zip_code(text)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting location with zip: {str(e)}")
            return result
    
    def parse_location_components(self, location: str) -> Dict[str, Optional[str]]:
        """
        Parse location string into components
        """
        result = {
            'city': None,
            'state': None,
            'zip_code': None
        }
        
        try:
            if not location:
                return result
            
            # Pattern: "City, ST ZIP" or "City, State ZIP"
            match = re.match(
                r'([^,]+),\s*([A-Z]{2}|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*(\d{5}(?:-\d{4})?)?',
                location.strip()
            )
            
            if match:
                city = match.group(1).strip()
                state = match.group(2).strip()
                zip_code = match.group(3).strip() if match.group(3) else None
                
                # Normalize
                state_normalized = self._normalize_state(state)
                city_clean = self._clean_city_name(city)
                
                if city_clean and state_normalized:
                    result['city'] = city_clean
                    result['state'] = state_normalized
                    result['zip_code'] = zip_code
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing location components: {str(e)}")
            return result
    
    def _normalize_state(self, state: str) -> Optional[str]:
        """Normalize state name to 2-letter abbreviation"""
        if not state:
            return None
        
        state = state.strip()
        
        # Already an abbreviation
        if len(state) == 2 and state.upper() in self.us_states:
            return state.upper()
        
        # Full state name
        state_lower = state.lower()
        if state_lower in self.state_name_to_abbr:
            return self.state_name_to_abbr[state_lower]
        
        return None
    
    def _clean_city_name(self, city: str) -> Optional[str]:
        """Clean and validate city name with CSV-driven junk filtering and US-only validation"""
        if not city:
            return None
        
        # Remove extra whitespace
        city = ' '.join(city.split())
        
        # Remove common location prefixes that get captured by regex
        # "Agent Santa Clara" → "Santa Clara"
        # "Engineer At Charlotte" → "Charlotte"
        # "Location Of Concord" → "Concord"
        location_prefixes = [
            r'^Agent\s+',
            r'^Engineer\s+At\s+',
            r'^Location\s+Of\s+',
            r'^Onsite\s+In\s+',
            r'^Based\s+In\s+',
            r'^Located\s+In\s+',
            r'^Ca\s+Or\s+',  # "Ca Or Austin" → "Austin"
            r'^Or\s+',        # "Or Dallas" → "Dallas"
            r'^And\s+',
            r'^At\s+',
            r'^In\s+',
            r'^Various\s+',   # "Various Product Lines" → reject later
        ]
        for prefix_pattern in location_prefixes:
            city = re.sub(prefix_pattern, '', city, flags=re.IGNORECASE)
        
        # Title case
        city = city.title()
        
        # Validate length (3-30 chars for city names)
        if len(city) < 3 or len(city) > 30:
            self.logger.debug(f"✗ Rejected location: {city} (invalid length: {len(city)})")
            return None
        
        # Must have letters
        if not any(c.isalpha() for c in city):
            return None
        
        city_lower = city.lower()
        
        # DYNAMIC VALIDATION: Pattern-based checks (NO hardcoded company lists!)
        
        # 1. BUSINESS SUFFIX PATTERN: Has company-like suffixes
        business_suffixes = ['inc', 'llc', 'corp', 'ltd', 'limited', 'corporation',
                             'solutions', 'technologies', 'systems', 'services', 
                             'consulting', 'group', 'partners', 'associates']
        if any(suffix in city_lower for suffix in business_suffixes):
            self.logger.debug(f"❌ Location has business suffix (likely company): {city}")
            return None
        
        # 2. CAMELCASE PATTERN: Internal capitals without spaces = company name
        # "TechCorp", "DataSystems" vs "Austin", "Boston"
        if len(city) > 1 and city[0].isupper():
            # Check for internal capitals (CamelCase)
            internal_caps = sum(1 for c in city[1:] if c.isupper())
            if internal_caps > 0 and ' ' not in city:
                self.logger.debug(f"❌ Location has CamelCase pattern (likely company): {city}")
                return None
        
        # 3. TECH ACRONYM PATTERN: All caps 2-4 letters (AI, ML, AWS, SQL)
        if city.isupper() and 2 <= len(city) <= 4:
            # Check if it's a valid state abbreviation
            if city.upper() not in self.state_abbreviations:
                self.logger.debug(f"❌ Location is tech acronym: {city}")
                return None
        
        # 4. HTML/ENCODING ARTIFACTS
        html_artifacts = ['&nbsp', '&amp', '&quot', '&lt', '&gt', '&#', '\u0026nbsp', 'nbsp', 'quot', 'amp']
        if any(artifact in city_lower for artifact in html_artifacts):
            self.logger.debug(f"❌ Location contains HTML entity: {city}")
            return None
        
        # 5. GENERIC SINGLE WORDS (common false positives)
        generic_words = ['area', 'story', 'team', 'group', 'department', 'division', 'unit', 'office', 'branch']
        if city_lower in generic_words:
            self.logger.debug(f"❌ Location is generic word: {city}")
            return None
        
        # 0. SEMANTIC VALIDATION - Reject common phrases, verbs, tech terms
        # Check if entire city name is a common phrase
        if city_lower in self.common_phrases:
            self.logger.debug(f"✗ Rejected common phrase: {city}")
            return None
        
        # Check if it's a verb/adjective
        if city_lower in self.common_verbs_adjectives:
            self.logger.debug(f"✗ Rejected verb/adjective: {city}")
            return None
        
        # Check if it's a technology term
        if city_lower in self.tech_terms:
            self.logger.debug(f"✗ Rejected technology term: {city}")
            return None
        
        # Check if starts with invalid prefix (Or, And, For, etc.)
        first_word = city_lower.split()[0] if ' ' in city_lower else city_lower
        if first_word in self.invalid_prefixes:
            self.logger.debug(f"✗ Rejected location starting with '{first_word}': {city}")
            return None
        
        # 1. Check for street name indicators (road, street, avenue, etc.)
        for indicator in self.street_name_indicators:
            if indicator in city_lower:
                self.logger.debug(f"✗ Rejected street name: {city} (contains '{indicator}')")
                return None
        
        # 2. Check against CSV-loaded false positives
        for fp in self.location_false_positives:
            if fp in city_lower:
                self.logger.debug(f"✗ Rejected junk location: {city} (contains '{fp}' from CSV)")
                return None
        
        # 3. Check against junk patterns (sentence fragments, verbs, etc.)
        for pattern in self.location_junk_patterns:
            if pattern.search(city_lower):
                self.logger.debug(f"✗ Rejected junk location: {city} (matches junk pattern)")
                return None
        
        # 4. Reject if it's mostly non-alphabetic
        alpha_count = sum(c.isalpha() or c.isspace() for c in city)
        if alpha_count / len(city) < 0.7:
            self.logger.debug(f"✗ Rejected junk location: {city} (too many non-alpha chars)")
            return None
        
        # 5. Dynamic Heuristics - Word count and capitalization
        words = city.split()
        if len(words) >= 3:
            # If it's 3+ words, it MUST be in our major cities list (e.g. Salt Lake City, San Francisco)
            # or it's likely a sentence fragment like "Applicable Privacy Rights"
            if city_lower not in self.us_major_cities:
                self.logger.debug(f"✗ Rejected multi-word non-city: {city}")
                return None
        
        # Check if the city contains common lowercase connectors (and, or, with) 
        # which signify a sentence fragment that regex accidentally captured.
        if any(connector in city.split() for connector in ['or', 'and', 'with', 'from']):
            # If it has "or" (lowercase), it's definitely a sentence part
            self.logger.debug(f"✗ Rejected sentence fragment with connector: {city}")
            return None

        # 6. US-only validation: Check if city is in major US cities list
        # This is optional but improves quality - only validate if we have the list
        if self.us_major_cities:
            if city_lower not in self.us_major_cities:
                # Allow if it's a less common single/double word city, but log it
                self.logger.debug(f"⚠ Location not in major US cities list: {city} (may be valid but uncommon)")
        
        return city
    
    def _is_valid_us_zip(self, zip_code: str) -> bool:
        """Validate US ZIP code"""
        if not zip_code:
            return False
        
        # Must be 5 digits or 5+4 format
        if not re.match(r'^\d{5}(?:-\d{4})?$', zip_code):
            return False
        
        return True
