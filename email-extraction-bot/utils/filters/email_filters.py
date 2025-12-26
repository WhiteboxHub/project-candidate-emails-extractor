import re
import os
import joblib
import logging
from typing import Dict, List, Optional
from email.utils import parseaddr
import mysql.connector

logger = logging.getLogger(__name__)


class EmailFilter:
    """
    DB-driven email filtering engine - NO HARDCODED FILTERS.
    - Priority-based allow/block rules from database
    - Content scoring using DB weights & targets
    - Validates ALL email addresses (sender, extracted, etc.)
    - Optional ML classifier
    - Calendar invite support
    """

    def __init__(self, config: dict, db_config: dict):
        self.config = config
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)

        # DB-loaded rules (priority ordered)
        self.sender_rules: List[Dict] = []  # For sender filtering
        self.email_rules: List[Dict] = []  # For extracted email validation
        self.content_rules: List[Dict] = []  # For content scoring
        self.name_rules: List[Dict] = []  # For name validation
        self.company_rules: List[Dict] = []  # For company validation

        # ML classifier
        self.use_ml = config.get("filters", {}).get("use_ml_classifier", False)
        self.classifier = None
        self.vectorizer = None

        # Load rules from DB
        self._load_rules_from_db()

        if self.use_ml:
            self._load_ml_model()

    # --------------------------------------------------
    # DB RULE LOADING
    # --------------------------------------------------
    def _load_rules_from_db(self):
        """Load all rules from database - NO HARDCODED VALUES"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(
                """
                SELECT category, keywords, match_type, action, priority, 
                       COALESCE(weight, 1) as weight, 
                       COALESCE(target, 'both') as target
                FROM job_automation_keywords
                WHERE is_active = 1 AND source = 'email_extractor'
                ORDER BY priority ASC
                """
            )
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            for row in rows:
                keywords = [
                    k.strip().lower() for k in row["keywords"].split(",") if k.strip()
                ]

                # Compile regex if needed
                if row["match_type"] == "regex":
                    try:
                        keywords = [re.compile(k, re.IGNORECASE) for k in keywords]
                    except re.error as e:
                        self.logger.warning(f"Invalid regex in {row['category']}: {e}")
                        continue

                rule = {
                    "category": row["category"],
                    "match_type": row["match_type"],
                    "action": row["action"],
                    "priority": row["priority"],
                    "keywords": keywords,
                    "weight": row["weight"],
                    "target": row["target"],
                }

                # Separate rules by type
                if row["category"] in ["recruiter_keywords", "anti_recruiter_keywords"]:
                    self.content_rules.append(rule)
                elif row["category"].startswith("invalid_name_"):
                    self.name_rules.append(rule)
                elif row["category"].startswith("invalid_company_"):
                    self.company_rules.append(rule)
                else:
                    # All other rules apply to both sender and extracted emails
                    self.sender_rules.append(rule)
                    self.email_rules.append(rule)

            self.logger.info(
                "Loaded %d sender/email rules, %d content rules, %d name rules, %d company rules from DB",
                len(self.sender_rules),
                len(self.content_rules),
                len(self.name_rules),
                len(self.company_rules),
            )

        except Exception as e:
            self.logger.error(f"Failed to load DB rules: {e}")
            raise

    # --------------------------------------------------
    # ML SUPPORT
    # --------------------------------------------------
    def _load_ml_model(self):
        try:
            model_dir = self.config.get("filters", {}).get("ml_model_dir", "../models")
            classifier_path = os.path.join(model_dir, "classifier.pkl")
            vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

            if os.path.exists(classifier_path) and os.path.exists(vectorizer_path):
                self.classifier = joblib.load(classifier_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.logger.info("ML classifier loaded")
            else:
                self.logger.warning("ML model files not found, disabling ML")
                self.use_ml = False
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            self.use_ml = False

    # --------------------------------------------------
    # UTILS
    # --------------------------------------------------
    def _extract_email(self, from_header: str) -> Optional[str]:
        """Extract email address from header"""
        if not from_header:
            return None
        _, email = parseaddr(from_header)
        email = email.lower().strip()
        return email if "@" in email else None

    def _rule_matches(self, email: str, rule: Dict) -> bool:
        """
        Check if email matches a rule - handles exact, contains, and regex
        """
        if not email or "@" not in email:
            return False

        email_lower = email.lower()
        local_part = email.split("@")[0].lower()
        domain = email.split("@")[1].lower()

        for kw in rule["keywords"]:
            if rule["match_type"] == "exact":
                # Check exact match against full email, domain, or local part
                if email_lower == kw or domain == kw or local_part == kw:
                    return True
                # Also check if keyword is contained in local part (for system emails like "donotreply")
                # This catches variations like "donotreply", "do-not-reply", etc.
                if kw in local_part:
                    return True
            elif rule["match_type"] == "contains":
                # Check if keyword is contained in email (full email, local part, or domain)
                if isinstance(kw, str):
                    if kw in email_lower or kw in local_part or kw in domain:
                        return True
            elif rule["match_type"] == "regex":
                # Regex match against full email, local part, and domain
                if isinstance(kw, re.Pattern):
                    if kw.search(email_lower) or kw.search(local_part) or kw.search(domain):
                        return True

        return False

    # --------------------------------------------------
    # EMAIL VALIDATION (for extracted emails)
    # --------------------------------------------------
    def is_email_allowed(self, email: str) -> bool:
        """
        Validate extracted email against DB rules.
        Returns True if email should be allowed, False if blocked.
        
        Priority order:
        1. Allowlist rules (priority 1-2) - if match, allow
        2. Blocklist rules (priority 50-100) - if match, block
        3. Default: allow (if no rules match)
        """
        if not email or "@" not in email:
            return False

        email_lower = email.lower().strip()

        # Process rules in priority order (already sorted)
        for rule in self.email_rules:
            if self._rule_matches(email_lower, rule):
                action = rule["action"]
                self.logger.debug(
                    f"Email {email_lower} → {action} ({rule['category']}, priority {rule['priority']})"
                )
                return action == "allow"

        # Default: allow if no rules match
        return True

    def is_email_blocked(self, email: str) -> bool:
        """Check if email is blocked - inverse of is_email_allowed"""
        return not self.is_email_allowed(email)

    # --------------------------------------------------
    # SENDER FILTER
    # --------------------------------------------------
    def check_sender(self, email_data: Dict) -> str:
        """
        Check sender email against DB rules.
        Returns 'allow' or 'block' based on priority-ordered rules.
        """
        msg = email_data.get("message") if isinstance(email_data, dict) else email_data
        from_header = msg.get("From", "") if hasattr(msg, "get") else getattr(msg, "From", "")
        email = self._extract_email(from_header)
        
        if not email:
            return "block"

        # Process rules in priority order
        for rule in self.sender_rules:
            if self._rule_matches(email, rule):
                action = rule["action"]
                self.logger.debug(
                    f"Sender {email} → {action} ({rule['category']}, priority {rule['priority']})"
                )
                return action

        # Default: allow if no rules match
        return "allow"

    def is_junk_email(self, from_header: str) -> bool:
        """Legacy method - checks if sender is junk"""
        email = self._extract_email(from_header)
        if not email:
            return True
        return self.check_sender({"message": {"From": from_header}}) == "block"

    # --------------------------------------------------
    # CONTENT SCORING
    # --------------------------------------------------
    def score_content(self, subject: str, body: str) -> int:
        """Score email content using DB keyword weights"""
        score = 0
        subject_lower = (subject or "").lower()
        body_lower = (body or "").lower()

        for rule in self.content_rules:
            weight = rule.get("weight", 1)
            target = rule.get("target", "both")

            for kw in rule["keywords"]:
                if isinstance(kw, str):
                    hit = False
                    if target in ["subject", "both"] and kw in subject_lower:
                        hit = True
                    if target in ["body", "both"] and kw in body_lower:
                        hit = True
                    if hit:
                        score += weight
                        self.logger.debug(f"Content match: '{kw}' → +{weight} (score: {score})")

        return score

    # --------------------------------------------------
    # RECRUITER DETECTION
    # --------------------------------------------------
    def is_recruiter_email(self, subject: str, body: str, from_header: str) -> bool:
        """Check if email is from a recruiter"""
        if self.is_junk_email(from_header):
            return False

        # ML classifier
        if self.use_ml and self.classifier and self.vectorizer:
            try:
                features = self.vectorizer.transform([f"{subject} {body} {from_header}"])
                return self.classifier.predict(features)[0] == 1
            except Exception as e:
                self.logger.error(f"ML classification error: {e}")
                return False

        score = self.score_content(subject, body)
        threshold = self.config.get("filters", {}).get("content_score_threshold", 2)
        return score >= threshold

    # --------------------------------------------------
    # CALENDAR SUPPORT
    # --------------------------------------------------
    def is_calendar_invite(self, email_message) -> bool:
        """Check if email is a calendar invite"""
        try:
            if hasattr(email_message, "walk"):
                for part in email_message.walk():
                    if part.get_content_type() == "text/calendar":
                        return True
            return False
        except Exception:
            return False

    # --------------------------------------------------
    # PIPELINE HELPER
    # --------------------------------------------------
    def filter_emails(self, emails: List[Dict], cleaner) -> List[Dict]:
        """Filter emails using DB rules"""
        filtered = []

        for email_data in emails:
            try:
                msg = email_data["message"]
                from_header = msg.get("From", "")
                subject = msg.get("Subject", "")

                # Always include calendar invites if configured
                if self.config.get("processing", {}).get("calendar_invites", {}).get("process", True):
                    if self.is_calendar_invite(msg):
                        filtered.append(email_data)
                        continue

                if self.is_junk_email(from_header):
                    continue

                body = cleaner.extract_body(msg)

                if self.is_recruiter_email(subject, body, from_header):
                    email_data["clean_body"] = body
                    filtered.append(email_data)

            except Exception as e:
                self.logger.error(f"Error filtering email: {e}")

        self.logger.info("Filtered %d emails from %d total", len(filtered), len(emails))
        return filtered

    # --------------------------------------------------
    # NAME AND COMPANY VALIDATION (DB RULES)
    # --------------------------------------------------
    def _check_name_against_db_rules(self, name: str) -> bool:
        """
        Check if name is valid using DB rules.
        Returns True if name should be allowed, False if blocked.
        """
        if not name or len(name.strip()) < 2:
            return False
        
        name_lower = name.lower().strip()
        
        # Check against invalid name rules
        for rule in self.name_rules:
            if self._rule_matches_text(name_lower, rule):
                self.logger.debug(
                    f"Name '{name_lower}' blocked by rule: {rule['category']}"
                )
                return False
        
        return True
    
    def _check_company_against_db_rules(self, company: str) -> bool:
        """
        Check if company is valid using DB rules.
        Returns True if company should be allowed, False if blocked.
        """
        if not company or len(company.strip()) < 2:
            return False
        
        company_lower = company.lower().strip()
        
        # Check against invalid company rules
        for rule in self.company_rules:
            if self._rule_matches_text(company_lower, rule):
                self.logger.debug(
                    f"Company '{company_lower}' blocked by rule: {rule['category']}"
                )
                return False
        
        return True
    
    def _rule_matches_text(self, text: str, rule: Dict) -> bool:
        """
        Check if text matches a rule - handles exact, contains, and regex for text validation.
        Similar to _rule_matches but for text fields (name, company) instead of emails.
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        for kw in rule["keywords"]:
            if rule["match_type"] == "exact":
                if isinstance(kw, str) and text_lower == kw:
                    return True
            elif rule["match_type"] == "contains":
                if isinstance(kw, str) and kw in text_lower:
                    return True
            elif rule["match_type"] == "regex":
                if isinstance(kw, re.Pattern):
                    if kw.search(text_lower):
                        return True
                elif isinstance(kw, str):
                    try:
                        pattern = re.compile(kw, re.IGNORECASE)
                        if pattern.search(text_lower):
                            return True
                    except re.error:
                        pass
        
        return False

    # --------------------------------------------------
    # PHONE NUMBER VALIDATION (DB RULES)
    # --------------------------------------------------
    def _is_valid_phone(self, phone: str) -> bool:
        """
        Validate phone number using DB rules from job_automation_keywords table.
        Returns True if phone should be allowed, False if blocked.
        """
        if not phone or len(phone.strip()) < 7:
            return False
        
        phone_clean = phone.strip()
        
        # Check against invalid phone rules from DB
        for rule in self.email_rules:  # Reuse email_rules since they apply to 'both'
            if rule['category'].startswith('invalid_phone_'):
                if self._rule_matches_text(phone_clean, rule):
                    self.logger.debug(
                        f"Phone '{phone_clean}' blocked by rule: {rule['category']}"
                    )
                    return False
        
        # Basic format validation
        if not phone_clean.startswith('+'):
            return False
        
        # Must have at least 10 digits
        digits = ''.join(c for c in phone_clean if c.isdigit())
        if len(digits) < 10:
            return False
        
        return True

    # --------------------------------------------------
    # LINKEDIN ID VALIDATION (DB RULES)
    # --------------------------------------------------
    def _is_valid_linkedin(self, linkedin_id: str) -> bool:
        """
        Validate LinkedIn ID using DB rules from job_automation_keywords table.
        Returns True if LinkedIn ID should be allowed, False if blocked.
        """
        if not linkedin_id or len(linkedin_id.strip()) < 3:
            return False
        
        linkedin_clean = linkedin_id.lower().strip()
        
        # Check against invalid LinkedIn rules from DB
        for rule in self.email_rules:  # Reuse email_rules since they apply to 'both'
            if rule['category'].startswith('invalid_linkedin_'):
                if self._rule_matches_text(linkedin_clean, rule):
                    self.logger.debug(
                        f"LinkedIn ID '{linkedin_clean}' blocked by rule: {rule['category']}"
                    )
                    return False
        
        # Basic format validation
        if len(linkedin_id) > 100:
            return False
        
        if linkedin_id.count(' ') >= 3:
            return False
        
        return True

    # --------------------------------------------------
    # LOCATION VALIDATION (DB RULES)
    # --------------------------------------------------
    def _is_valid_location(self, location: str) -> bool:
        """
        Validate location using DB rules from job_automation_keywords table.
        Returns True if location should be allowed, False if blocked.
        """
        if not location or len(location.strip()) < 2:
            return False
        
        location_clean = location.lower().strip()
        
        # Check against invalid location rules from DB
        for rule in self.email_rules:  # Reuse email_rules since they apply to 'both'
            if rule['category'].startswith('invalid_location_'):
                if self._rule_matches_text(location_clean, rule):
                    self.logger.debug(
                        f"Location '{location_clean}' blocked by rule: {rule['category']}"
                    )
                    return False
        
        # Basic format validation
        if len(location) > 100:
            return False
        
        if len(location.split()) > 6:
            return False
        
        return True

    # --------------------------------------------------
    # RULE STATISTICS
    # --------------------------------------------------
    def get_rule_statistics(self) -> Dict:
        """
        Get statistics about loaded rules for debugging and monitoring.
        Returns a dictionary with rule counts by category and type.
        """
        stats = {
            'total_sender_rules': len(self.sender_rules),
            'total_email_rules': len(self.email_rules),
            'total_content_rules': len(self.content_rules),
            'total_name_rules': len(self.name_rules),
            'total_company_rules': len(self.company_rules),
            'allow_rules': 0,
            'block_rules': 0,
            'by_category': {}
        }
        
        # Count all rules
        all_rules = (
            self.sender_rules + 
            self.email_rules + 
            self.content_rules + 
            self.name_rules + 
            self.company_rules
        )
        
        for rule in all_rules:
            # Count by action
            if rule['action'] == 'allow':
                stats['allow_rules'] += 1
            elif rule['action'] == 'block':
                stats['block_rules'] += 1
            
            # Count by category
            category = rule['category']
            if category not in stats['by_category']:
                stats['by_category'][category] = 0
            stats['by_category'][category] += 1
        
        return stats
