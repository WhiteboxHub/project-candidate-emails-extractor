import re
import logging
from typing import Optional, List, Dict
import spacy
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)


class PositionExtractor:
    """Extract job positions from email text using multiple methods"""
    
    def __init__(self, spacy_model=None):
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model if provided
        self.nlp = spacy_model
        if not self.nlp:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except:
                self.logger.warning("SpaCy model not loaded - spacy extraction will be disabled")
        
        self.filter_repo = get_filter_repository()
        self.job_title_keywords = self._load_job_title_keywords()
        
        # Load junk intro phrases (from CSV)
        self.junk_intro_phrases = []
        
        # Load filter repository for CSV-driven configuration
        self._load_position_filters()
        
        # COMPREHENSIVE Regex patterns for job position extraction
        # Ordered by specificity (most specific first)
        # RELAXED: Allow _ at start of patterns to capture "_Senior Developer" etc.
        self.position_patterns = [
            # SUBJECT LINE PATTERNS (very common in recruiter emails)
            
            # "Technical Interview - GenAI Architect" or "Interview for Senior Engineer"
            r'(?:technical\s+)?(?:interview|screening|discussion)(?:\s*[-–—_]\s*|\s+for\s+)([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|SDET|QA))',
            
            # "Job Google ADK AI Engineer is shared with you" or "Job AI Engineer is shared"
            r'(?:job|position|role)\s+(?:[A-Za-z0-9]+\s+){0,4}([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|Programmer|Tester|Scientist|Researcher|SRE|DevOps|SDET|QA))\s+is\s+shared',
            
            # "Urgent Requirement || Google ADK AI Engineer ||" or "W2/ C2C || Lead Java Developer ||"
            r'(?:requirement|opening|opportunity|position|role|job)\s*\|\|?\s*(?:[A-Za-z0-9&]+\s+){0,3}([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|Programmer|Tester|Scientist|Researcher|SRE|DevOps|SDET|QA))',
            
            # "Gen AI Engineer Irving,TX" or "GenAI Architect - Charlotte"
            r'^([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|SRE|DevOps|SDET|QA))\s*[-–—,]\s*[A-Z][a-z]+',
            
            # "Fulltime Software Engineering Director Job at Dallas"
            r'\b(?:fulltime|full-time|part-time|contract)?\s*([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|SRE|DevOps|SDET|QA))\s+(?:job|position|role|opening)',
            
            # "Looking for Gen AI Engineer /Lead" or "seeking GenAI Architect"
            r'(?:looking for|seeking|hiring|need|require|searching for)\s+(?:a\s+|an\s+)?([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Administrator|Coordinator|Lead|Director|Programmer|Tester|Scientist|Researcher|SDET|QA))',
            
            # "JOB ROLE for Agentic AI Engineer" or "opening for AI Agent Developer"
            r'(?:job\s+role|opening|vacancy|position)\s+for\s+(?:a\s+|an\s+)?([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|SDET|QA))',
            
            # "Lead Java Developer || Charlotte" or "Gen AI Lead || Dallas"
            r'\|\|?\s*([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director))\s*\|\|',
            
            # "Senior AI Engineer Opportunity" or "ML Research Engineering role"
            r'([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|Researcher))\s+(?:opportunity|role|position|opening)',
            
            # "ML Ops Senior Engineer ::" or "AI/ML Engineer with Java"
            r'([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director))\s*(?:::|with|and)',
            
            # "Immediate Hire - AI/ML Engineer" or "Urgent hiring - GenAI Architect"
            r'(?:immediate|urgent|hot)\s+(?:hire|hiring|requirement|opening)\s*[-–—]\s*([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director))',
            
            # "Direct Client job opportunity for AI Agent Developer"
            r'(?:job\s+)?opportunity\s+for\s+([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director))',
            
            # "Onsite Confirmation : MLOPs + AI Engineer"
            r'(?:confirmation|requirement)\s*:\s*([A-Z_][\w\s/\-\.&+]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director))',
            
            # "We have urgent open position for Generative AI Engineer"
            r'(?:open\s+)?position\s+for\s+([A-Z_][\w\s/\-\.&]+?(?:Developer|Engineer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director))',
            
            # Broad fallback - any capitalized phrase with job keywords
            r'\b([A-Z_][\w\s/\-\.&]*?(?:AI|ML|Data|Cloud|DevOps|Software|Full\s+Stack|Backend|Frontend|Mobile|Gen\s*AI|Machine\s+Learning|Agentic)[\w\s/\-\.&]*?(?:Engineer|Developer|Architect|Manager|Analyst|Designer|Consultant|Specialist|Lead|Director|Scientist|Researcher|SDET|QA))\b',
        ]
        
        # Common job title suffixes for validation - will be loaded from CSV
        self.job_title_suffixes = set()
        self.acronym_capitalizations = {}
    
    def _load_job_title_keywords(self) -> set:
        """Load job title keywords from filter repository (CSV)"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            if 'job_position_trigger_words' in keyword_lists:
                keywords = keyword_lists['job_position_trigger_words']
                job_titles = {kw.lower().strip() for kw in keywords}
                self.logger.info(f"✓ Loaded {len(job_titles)} job position trigger words from CSV")
                return job_titles
            else:
                self.logger.warning("⚠ job_position_trigger_words not found in CSV - using empty set")
                return set()
                
        except Exception as e:
            self.logger.error(f"Failed to load job title keywords from CSV: {str(e)}")
            return set()
    
    def _load_position_filters(self):
        """Load position cleaning filters from CSV"""
        try:
            keyword_lists = self.filter_repo.get_keyword_lists()
            
            # Load marketing/fluff words
            if 'position_marketing_words' in keyword_lists:
                self.marketing_words = [
                    kw.strip() for kw in keyword_lists['position_marketing_words']
                ]
                self.logger.info(f"✓ Loaded {len(self.marketing_words)} position marketing words from CSV")
            else:
                self.marketing_words = []
                self.logger.warning("⚠ position_marketing_words not found in CSV")
            
            # Load prefixes to remove
            if 'position_prefixes_remove' in keyword_lists:
                self.prefixes_to_remove = [
                    kw.strip() for kw in keyword_lists['position_prefixes_remove']
                ]
                self.logger.info(f"✓ Loaded {len(self.prefixes_to_remove)} position prefixes from CSV")
            else:
                self.prefixes_to_remove = []
                self.logger.warning("⚠ position_prefixes_remove not found in CSV")
            
            # Load trailing artifacts
            if 'position_trailing_artifacts' in keyword_lists:
                self.trailing_artifacts = [
                    kw.strip() for kw in keyword_lists['position_trailing_artifacts']
                ]
                self.logger.info(f"✓ Loaded {len(self.trailing_artifacts)} position trailing artifacts from CSV")
            else:
                self.trailing_artifacts = []
                self.logger.warning("⚠ position_trailing_artifacts not found in CSV")
            
            # Load HTML tag patterns
            if 'html_tag_patterns' in keyword_lists:
                self.html_patterns = [
                    re.compile(pattern.strip(), re.IGNORECASE)
                    for pattern in keyword_lists['html_tag_patterns']
                ]
                self.logger.info(f"✓ Loaded {len(self.html_patterns)} HTML tag patterns from CSV")
            else:
                # Fallback to basic HTML tag pattern
                self.html_patterns = [re.compile(r'<[^>]*>', re.IGNORECASE)]
                self.logger.warning("⚠ html_tag_patterns not found in CSV - using fallback")
            
            # Load job title suffixes
            if 'job_title_suffixes' in keyword_lists:
                self.job_title_suffixes = set(
                    kw.lower().strip() for kw in keyword_lists['job_title_suffixes']
                )
                self.logger.info(f"✓ Loaded {len(self.job_title_suffixes)} job title suffixes from CSV")
            else:
                self.job_title_suffixes = set()
                self.logger.warning("⚠ job_title_suffixes not found in CSV")
            
            # Load acronym capitalizations (format: "ai|AI,ml|ML")
            if 'acronym_capitalizations' in keyword_lists:
                self.acronym_capitalizations = {}
                for mapping in keyword_lists['acronym_capitalizations']:
                    if '|' in mapping:
                        lowercase, proper = mapping.split('|', 1)
                        self.acronym_capitalizations[lowercase.lower().strip()] = proper.strip()
                self.logger.info(f"✓ Loaded {len(self.acronym_capitalizations)} acronym capitalizations from CSV")
            else:
                self.acronym_capitalizations = {}
                self.logger.warning("⚠ acronym_capitalizations not found in CSV")
                
            # Load junk intro phrases
            if 'position_junk_intro_phrases' in keyword_lists:
                self.junk_intro_phrases = [
                    kw.lower().strip() for kw in keyword_lists['position_junk_intro_phrases']
                ]
                self.logger.info(f"✓ Loaded {len(self.junk_intro_phrases)} junk intro phrases from CSV")
            else:
                self.junk_intro_phrases = []
                self.logger.warning("⚠ position_junk_intro_phrases not found in CSV")
            # NEW: Load recruiter titles to block
            if 'blocked_recruiter_titles' in keyword_lists:
                self.recruiter_titles = set(
                    kw.lower().strip() for kw in keyword_lists['blocked_recruiter_titles']
                )
                self.logger.info(f"✓ Loaded {len(self.recruiter_titles)} recruiter titles from CSV")
            else:
                self.recruiter_titles = set()
                self.logger.warning("⚠ blocked_recruiter_titles not found in CSV")
            
            # NEW: Load company prefixes to remove
            if 'position_company_prefixes' in keyword_lists:
                self.company_prefixes = [
                    kw.lower().strip() for kw in keyword_lists['position_company_prefixes']
                ]
                self.logger.info(f"✓ Loaded {len(self.company_prefixes)} company prefixes from CSV")
            else:
                self.company_prefixes = []
                self.logger.warning("⚠ position_company_prefixes not found in CSV")
            
            # NEW: Load core job keywords (required for valid positions)
            if 'position_core_keywords' in keyword_lists:
                self.core_keywords = set(
                    kw.lower().strip() for kw in keyword_lists['position_core_keywords']
                )
                self.logger.info(f"✓ Loaded {len(self.core_keywords)} core job keywords from CSV")
            else:
                self.core_keywords = set()
                self.logger.warning("⚠ position_core_keywords not found in CSV")
            
            # NEW: Load marketing fluff phrases
            if 'position_marketing_fluff' in keyword_lists:
                self.marketing_fluff = [
                    kw.lower().strip() for kw in keyword_lists['position_marketing_fluff']
                ]
                self.logger.info(f"✓ Loaded {len(self.marketing_fluff)} marketing fluff phrases from CSV")
            else:
                self.marketing_fluff = []
                self.logger.warning("⚠ position_marketing_fluff not found in CSV")
                
        except Exception as e:
            self.logger.error(f"Failed to load position filters from CSV: {str(e)}")
            # CRITICAL: Initialize ALL attributes to prevent AttributeError
            self.marketing_words = []
            self.prefixes_to_remove = []
            self.trailing_artifacts = []
            self.html_patterns = [re.compile(r'<[^>]*>', re.IGNORECASE)]
            self.job_title_suffixes = set()
            self.acronym_capitalizations = {}
            self.junk_intro_phrases = []
            self.recruiter_titles = set()
            self.company_prefixes = []
            self.core_keywords = set()
            self.marketing_fluff = []
    
    def _normalize_acronyms_in_text(self, text: str) -> str:
        """Normalize common acronym patterns BEFORE extraction
        
        This fixes issues where regex captures "I/ML" instead of "AI/ML"
        because it starts matching at the first capital letter 'I'.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized acronyms
        """
        if not text:
            return text
        
        # Common acronym patterns that get truncated or captured partially
        acronym_fixes = [
            (r'\bi/ml\b', 'AI/ML'),
            (r'\bi/nlp\b', 'AI/NLP'),
            (r'\bi/llm\b', 'AI/LLM'),
            (r'\bi/rag\b', 'AI/RAG'),
            (r'\bi engineer\b', 'AI Engineer'),
            (r'\bi architect\b', 'AI Architect'),
            (r'\bi developer\b', 'AI Developer'),
            (r'\bi specialist\b', 'AI Specialist'),
            (r'\bi consultant\b', 'AI Consultant'),
            (r'\bi leads\b', 'AI Lead'),
            (r'\bi agent\b', 'AI Agent'),
            (r'\bi automation\b', 'AI Automation'),
            (r'\ba i\b', 'AI'),
            (r'\bm l\b', 'ML'),
            (r'\bgen i\b', 'Gen AI'),
            (r'\bgen i/', 'Gen AI/'),
            (r'\bgen-i\b', 'Gen AI'),
            (r'\bgenai\b', 'GenAI'),
            (r'\bgenerative ai\b', 'Generative AI'),
            (r'\bagentic ai\b', 'Agentic AI'),
            (r'\bagentic\b', 'Agentic'),
            (r'\bgentic\b', 'Agentic'),  # FIX: Common typo "Gentic" → "Agentic"
        ]
        
        for pattern, replacement in acronym_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def extract_job_position_regex(self, text: str) -> Optional[str]:
        """
        Extract job position using regex patterns
        
        Args:
            text: Email body or subject text (already normalized for acronyms)
            
        Returns:
            Extracted job position or None
        """
        try:
            if not text:
                return None
            
            # NOTE: Acronym normalization is now done at entry point in extractor.py
            # No need to normalize here
            
            # Clean subject line of prefixes if it's a subject
            original_text = text
            text = self._clean_subject_prefixes(text)
            
            # Try each pattern
            for pattern in self.position_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                
                for match in matches:
                    position = match.group(1).strip()
                    
                    # Clean up the position
                    position = self._clean_position(position)
                    
                    # Validate
                    if self._is_valid_position(position):
                        self.logger.debug(f"✓ Regex extracted position: {position}")
                        return position
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in regex position extraction: {str(e)}")
            return None
    
    def extract_job_position_spacy(self, text: str) -> Optional[str]:
        """
        Extract job position using spaCy noun phrase extraction with trigger words
        
        Args:
            text: Email body text
            
        Returns:
            Extracted job position or None
        """
        try:
            if not self.nlp or not text:
                return None
            
            # Process text with spaCy
            doc = self.nlp(text[:2000])  # Limit to first 2000 chars for performance
            
            candidates = []
            
            # Extract noun phrases that contain job title trigger words
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                chunk_lower = chunk_text.lower()
                
                # Check if chunk contains any trigger word
                has_trigger = any(trigger in chunk_lower for trigger in self.job_title_keywords)
                
                if has_trigger:
                    # Clean and validate
                    position = self._clean_position(chunk_text)
                    
                    if self._is_valid_position(position):
                        # Calculate confidence based on trigger word match and position in text
                        confidence = self._calculate_confidence(position, chunk.start_char, len(text))
                        candidates.append({
                            'position': position,
                            'confidence': confidence,
                            'method': 'spacy_noun_phrase',
                            'source': 'body'
                        })
            
            # Return highest confidence candidate
            if candidates:
                candidates.sort(key=lambda x: x['confidence'], reverse=True)
                best = candidates[0]
                self.logger.debug(f"✓ SpaCy extracted position: {best['position']} (confidence: {best['confidence']:.2f})")
                return best['position']
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in spacy position extraction: {str(e)}")
            return None
    
    def extract_all_job_positions(self, text: str, subject: str = None) -> List[Dict]:
        """
        Extract all job positions from text with metadata
        
        Args:
            text: Email body text
            subject: Email subject line
            
        Returns:
            List of dicts with {job_position, method, confidence, source}
        """
        positions = []
        seen = set()
        
        try:
            # 1. Try subject line first (often contains job title) - HIGHEST PRIORITY
            if subject:
                subject_position = self.extract_job_position_regex(subject)
                if subject_position:
                    position_key = subject_position.lower()
                    if position_key not in seen:
                        positions.append({
                            'job_position': subject_position,
                            'method': 'regex',
                            'confidence': 0.95,  # VERY high confidence for subject line
                            'source': 'subject'
                        })
                        seen.add(position_key)
                        self.logger.info(f"🎯 Extracted position from SUBJECT: {subject_position}")
            
            # 2. Try regex on body
            body_position_regex = self.extract_job_position_regex(text)
            if body_position_regex:
                position_key = body_position_regex.lower()
                if position_key not in seen:
                    positions.append({
                        'job_position': body_position_regex,
                        'method': 'regex',
                        'confidence': 0.80,
                        'source': 'body'
                    })
                    seen.add(position_key)
            
            # 3. Try spacy on body
            body_position_spacy = self.extract_job_position_spacy(text)
            if body_position_spacy:
                position_key = body_position_spacy.lower()
                if position_key not in seen:
                    positions.append({
                        'job_position': body_position_spacy,
                        'method': 'spacy',
                        'confidence': 0.70,
                        'source': 'body'
                    })
                    seen.add(position_key)
            
            # Sort by confidence
            positions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return positions
        except Exception as e:
            self.logger.error(f"Error extracting all positions: {str(e)}")
            return positions
    
    def _clean_subject_prefixes(self, text: str) -> str:
        """Remove common subject line prefixes like RE:, FW:, and encrypted warnings"""
        if not text:
            return text
            
        # Remove [WARNING: MESSAGE ENCRYPTED], etc.
        text = re.sub(r'\[WARNING:\s*MESSAGE\s*ENCRYPTED\]', '', text, flags=re.IGNORECASE)
        # Remove RE:, FW:, etc.
        text = re.sub(r'^(?:RE|FW|Fwd|Automatic reply|Out of Office):\s*', '', text, flags=re.IGNORECASE)
        
        # Clean extra brackets/punctuation often at start
        text = re.sub(r'^[\[\]\-\|\s\:\!\?]+', '', text)
        
        return text.strip()

    def _clean_position(self, position: str) -> str:
        """Clean and normalize job position text with CSV-driven filters"""
        if not position:
            return position
        
        # 1. Strip HTML/XML tags and entities using CSV patterns
        position = self._strip_html_comprehensive(position)
        
        # 1.3. Remove interview patterns (TCS Interview--, etc.) - UNIVERSAL pattern
        # This works for ANY company: "TCS Interview--", "Google Interview:", etc.
        position = re.sub(r'\b\w+\s+interview\s*[-:]+\s*', '', position, flags=re.IGNORECASE)
        
        # 1.5. Remove company-specific prefixes (from CSV) - IMPROVED
        position_lower = position.lower()
        for company_prefix in self.company_prefixes:
            if company_prefix in position_lower:
                # Find the prefix and remove it + everything before it
                # Use word boundaries to avoid partial matches
                idx = position_lower.find(company_prefix)
                if idx != -1:
                    # Remove everything up to and including the prefix
                    position = position[idx + len(company_prefix):].strip()
                    position_lower = position.lower()
                    # Remove leading separators (-, :, etc.)
                    position = re.sub(r'^[\s\-:;,\.]+', '', position)
                    position_lower = position.lower()
        
        # 2. Remove marketing/fluff words (from CSV)
        position_lower = position.lower()
        for fluff in self.marketing_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(fluff.lower()) + r'\b'
            position = re.sub(pattern, '', position, flags=re.IGNORECASE)
            
        # 2.3. Remove marketing fluff phrases (from CSV)
        position_lower = position.lower()
        for fluff_phrase in self.marketing_fluff:
            if fluff_phrase in position_lower:
                # Remove the entire phrase and everything before it
                pattern = r'.*?' + re.escape(fluff_phrase) + r'[\s.\-:]*'
                position = re.sub(pattern, '', position, flags=re.IGNORECASE)
                position_lower = position.lower()
            
        # 2.5. Aggressively remove junk intro phrases ("Hi my name is...", etc. from CSV)
        position_lower = position.lower()
        for intro in self.junk_intro_phrases:
            if intro in position_lower:
                # Find the intro phrase and everything before it, and replace with empty
                # We also need to remove the person's name that usually follows "my name is..."
                # Use a regex that catches "my name is [Name] and I am a [Job]"
                name_intro_pattern = r'.*?\b' + re.escape(intro) + r'\b\s*(?:[A-Z][a-z]+\s+){1,3}(?:and\s+)?(?:i\s+am\s+)?(?:working\s+as\s+)?(?:a\s+)?'
                new_position = re.sub(name_intro_pattern, '', position, flags=re.IGNORECASE)
                if new_position != position:
                    position = new_position
                else:
                    # Simple fallback: just strip the intro phrase if it's not a name pattern
                    position = re.sub(r'.*?\b' + re.escape(intro) + r'\b\s*', '', position, flags=re.IGNORECASE)
                position_lower = position.lower()

        # 3. Remove common prefixes (from CSV)
        position_lower = position.lower()
        for prefix in self.prefixes_to_remove:
            if position_lower.startswith(prefix.lower()):
                position = position[len(prefix):].strip()
                position_lower = position.lower()
        
        # 3.5. Remove additional contextual prefixes - UNIVERSAL patterns
        # These work for ANY company, ANY domain
        contextual_prefixes = [
            r'^\s*[-–—_\|\/]+\s*',  # Leading separators
            r'^For\s+(?:the\s+)?',
            r'^Job\s+Description\s*[-:]\s*',
            r'^Job\s+Role\s*[-:]\s*',
            r'^Job\s+Title\s*[-:]\s*',
            r'^Position\s*[-:]\s*',
            r'^Role\s*[-:]\s*',
            r'^Immediate\s+',
            r'^Urgent\s+',
            r'^Opening\s+for\s+',
            r'^Vacancy\s+for\s+',
            r'^Requirement\s+for\s+',
            r'^Hiring\s+for\s+',
            r'^\w+\s+(?:is\s+)?(?:seeking|looking\s+for)\s+',  # "Company is seeking", "XYZ looking for"
        ]
        for prefix_pattern in contextual_prefixes:
            position = re.sub(prefix_pattern, '', position, flags=re.IGNORECASE)
        
        # 4. Remove trailing artifacts (from CSV)
        for artifact in self.trailing_artifacts:
            # Remove if it's the last word
            pattern = r'\s+' + re.escape(artifact.lower()) + r'$'
            position = re.sub(pattern, '', position, flags=re.IGNORECASE)
        
        # 5. Remove trailing words like "position", "role", "opening"
        position = re.sub(r'\s+(?:position|role|opening|opportunity|job|requirement|vacancy)$', '', position, flags=re.IGNORECASE)
        
        # 6. Remove extra whitespace (including multiple spaces)
        position = ' '.join(position.split())
        
        # 7. Title case (but preserve acronyms later)
        position = position.title()
        
        # 8. Fix acronym capitalization (AI, ML, NLP, etc.)
        position = self._fix_acronym_capitalization(position)
        
        # 9. Handle common abbreviations
        position = position.replace('Sr.', 'Senior')
        position = position.replace('Jr.', 'Junior')
        position = position.replace('Mgr', 'Manager')
        
        # 10. Fix common patterns
        # "And" at the beginning (from removing "Highly Skilled And...")
        position = re.sub(r'^\s*(?:and|with|at|for|is)\s+', '', position, flags=re.IGNORECASE)
        
        # 11. Final cleanup: if it still has "Hi My Name Is" (due to case sensitivity or title casing)
        junk_remnants = [
            r'.*?My Name Is\s+(?:[A-Z][a-z]+\s+){1,3}(?:And\s+)?(?:I\s+Am\s+)?(?:A\s+)?',
            r'.*?I Am A\s+',
            r'.*?Is Shared With You.*',
        ]
        for remnant in junk_remnants:
            position = re.sub(remnant, '', position, flags=re.IGNORECASE)
        
        # 12. CRITICAL: Final acronym normalization AFTER all cleaning
        # This catches patterns like "I/Ml Engineer" → "AI/ML Engineer" that may have been
        # created by title casing or other cleaning steps
        position = self._normalize_acronyms_in_text(position)
        
        # Final trim
        position = position.strip()
        # Remove leading/trailing non-alphanumeric (except some)
        position = re.sub(r'^[^a-zA-Z0-9]+', '', position)
        position = re.sub(r'[^a-zA-Z0-9\)]+$', '', position)
        
        return position
    
    def _fix_acronym_capitalization(self, text: str) -> str:
        """Fix capitalization for common acronyms (AI, ML, NLP, etc.) from CSV"""
        if not text or not self.acronym_capitalizations:
            return text
        
        # Replace each acronym with proper capitalization
        # Use word boundaries to avoid partial matches
        for lowercase, proper in self.acronym_capitalizations.items():
            # Match whole words only
            pattern = r'\b' + re.escape(lowercase.title()) + r'\b'
            text = re.sub(pattern, proper, text, flags=re.IGNORECASE)
        
        return text
    
    def _strip_html_comprehensive(self, text: str) -> str:
        """Comprehensively strip HTML tags and entities using CSV patterns"""
        if not text:
            return text
        
        # Apply all HTML patterns from CSV
        for pattern in self.html_patterns:
            text = pattern.sub('', text)
        
        # Additional cleanup for common HTML entities not in patterns
        import html
        text = html.unescape(text)
        
        # Remove any remaining angle brackets (malformed tags)
        text = re.sub(r'[<>]', '', text)
        
        return text.strip()
    
    def _is_valid_position(self, position: str) -> bool:
        """
        Validate if extracted text is a valid job position using CSV-driven rules
        
        Args:
            position: Cleaned position string
            
        Returns:
            True if valid position, False otherwise
        """
        if not position:
            return False
        
        # Length check (3-60 chars) - UPDATED: max reduced from 100 to 60 to reject long sentences
        if len(position) < 3 or len(position) > 60:
            self.logger.debug(f"❌ Position length invalid ({len(position)} chars): {position}")
            return False
        
        position_lower = position.lower()
        
        # 1. REJECT: Questions (What, Why, How, etc.)
        question_indicators = ['what ', 'why ', 'how ', 'when ', 'where ', 'who ', 'which ', '?']
        if any(ind in position_lower for ind in question_indicators):
            self.logger.debug(f"❌ Position is a question: {position}")
            return False
        
        # 2. REJECT: Starts with "Re:" or "RE:" (email subject prefix)
        if position.startswith('Re ') or position.startswith('RE:') or position.startswith('Re:'):
            self.logger.debug(f"❌ Position starts with Re: {position}")
            return False
        
        # 3. REJECT: Company name patterns (ends with common suffixes)
        company_suffixes = [' Software', ' Inc', ' LLC', ' Corp', ' Ltd', ' Portal', "'S Candidate Portal", "'s Candidate Portal"]
        if any(position.endswith(suffix) for suffix in company_suffixes):
            self.logger.debug(f"❌ Position looks like company name: {position}")
            return False
        
        # 4. REJECT: Generic tech terms without role context
        generic_tech_terms = ['Cloud Environments', 'Tech Stack', 'Software Engineering']
        if position in generic_tech_terms:
            self.logger.debug(f"❌ Position is generic tech term: {position}")
            return False
        
        # 5. REJECT: Portal/system names
        portal_indicators = [' Portal', ' System', ' Platform', ' Dashboard']
        if any(ind in position for ind in portal_indicators):
            self.logger.debug(f"❌ Position contains portal/system indicator: {position}")
            return False
        
        # Must have at least one letter
        if not any(c.isalpha() for c in position):
            return False
        
        # Must not be all uppercase (likely acronym or company name)
        if position.isupper() and len(position) > 10:
            return False
        
        position_lower = position.lower()
        
        # CRITICAL FIX: Block recruiter name + title patterns
        # Pattern: "FirstName LastName Recruitment Consultant" or "FirstName Recruitment Consultant"
        # This catches patterns like "Ilir Alija Recruitment Consultant", "Rlinda Dobratiqi Senior Recruitment Consultant"
        recruiter_name_title_pattern = r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Senior\s+)?(?:Recruitment|Recruiter|Staffing|Talent|HR)\s+(?:Consultant|Specialist|Manager|Coordinator|Lead)'
        if re.match(recruiter_name_title_pattern, position, re.IGNORECASE):
            self.logger.debug(f"⊘ Rejected recruiter name+title pattern: {position}")
            return False
        
        # NEW: Block if it matches blocked recruiter titles (from CSV)
        if self.recruiter_titles:
            for recruiter_title in self.recruiter_titles:
                if recruiter_title in position_lower:
                    self.logger.debug(f"⊘ Rejected recruiter title: {position}")
                    return False
        
        # NEW: Word count validation (2-8 words max)
        # CRITICAL FIX: Allow 2-word positions if they contain core keywords
        # This fixes "Gen AI" being rejected
        words = position.split()
        word_count = len(words)
        
        if word_count < 2:
            return False
        
        if word_count > 8:
            return False
        
        # NEW: Require at least one core keyword (from CSV)
        # For 2-word positions, this is CRITICAL to avoid false positives
        if self.core_keywords:
            has_core_keyword = any(keyword in position_lower for keyword in self.core_keywords)
            if not has_core_keyword:
                self.logger.debug(f"⊘ Rejected (no core keyword): {position}")
                return False
        
        # NEW: Block marketing fluff phrases (from CSV)
        if self.marketing_fluff:
            for fluff in self.marketing_fluff:
                if fluff in position_lower:
                    self.logger.debug(f"⊘ Rejected marketing fluff: {position}")
                    return False
        
        # Check if it contains a job title suffix or trigger word
        has_suffix = any(suffix in position_lower for suffix in self.job_title_suffixes)
        has_trigger = any(trigger in position_lower for trigger in self.job_title_keywords)
        
        if not (has_suffix or has_trigger):
            return False
        
        # Filter out common false positives
        false_positives = [
            'team', 'department', 'company', 'organization', 'group',
            'please', 'thank', 'regards', 'sincerely', 'best',
            'email', 'phone', 'contact', 'address'
        ]
        
        if any(fp in position_lower for fp in false_positives):
            return False
        
        return True
    
    def _calculate_confidence(self, position: str, char_position: int, text_length: int) -> float:
        """
        Calculate confidence score for extracted position
        
        Args:
            position: Extracted position text
            char_position: Character position in text
            text_length: Total text length
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Bonus for position early in text (first 500 chars)
        if char_position < 500:
            confidence += 0.2
        elif char_position < 1000:
            confidence += 0.1
        
        # Bonus for having job title suffix
        position_lower = position.lower()
        if any(suffix in position_lower for suffix in self.job_title_suffixes):
            confidence += 0.15
        
        # Bonus for multiple trigger words
        trigger_count = sum(1 for trigger in self.job_title_keywords if trigger in position_lower)
        if trigger_count >= 2:
            confidence += 0.1
        elif trigger_count == 1:
            confidence += 0.05
        
        # Penalty for very long positions (likely false positive)
        if len(position) > 60:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))