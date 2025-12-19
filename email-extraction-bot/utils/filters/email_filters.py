import re
import os
import joblib
import logging
from typing import Dict, List
from email.utils import parseaddr

logger = logging.getLogger(__name__)


class EmailFilter:
    """
    Fully DB-driven email filtering engine.
    - No hardcoded domains/emails
    - Priority-based allow/block rules
    - Optional ML classifier
    - Calendar invite support
    """

    HUMAN_LOCAL_PATTERN = re.compile(
        r'^[a-z]+([._-][a-z]+){0,2}$',
        re.IGNORECASE
    )

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------
    def __init__(self, config: dict, db_conn):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # DB-loaded rules (priority ordered)
        self.rules: List[Dict] = []

        # Content classification (DB driven)
        self.recruiter_keywords: List[str] = []
        self.anti_recruiter_keywords: List[str] = []

        # Load filters from DB
        self._load_rules_from_db(db_conn)

        # ML classifier (optional)
        self.use_ml = config.get('filters', {}).get('use_ml_classifier', False)
        self.classifier = None
        self.vectorizer = None

        if self.use_ml:
            self._load_ml_model()

    # --------------------------------------------------
    # DB LOAD
    # --------------------------------------------------
    def _load_rules_from_db(self, conn):
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT category, keywords, match_type, action, priority
            FROM job_automation_keywords
            WHERE is_active = 1
            ORDER BY priority ASC
        """)

        rows = cursor.fetchall()
        cursor.close()

        for row in rows:
            keywords = [
                k.strip().lower()
                for k in row["keywords"].split(",")
                if k.strip()
            ]

            match_type = row["match_type"]
            action = row["action"]
            category = row["category"]

            # Content rules
            if category == "recruiter_keywords":
                self.recruiter_keywords.extend(keywords)
                continue

            if category == "anti_recruiter_keywords":
                self.anti_recruiter_keywords.extend(keywords)
                continue

            # Regex compilation
            if match_type == "regex":
                keywords = [re.compile(k, re.IGNORECASE) for k in keywords]

            self.rules.append({
                "category": category,
                "match_type": match_type,
                "action": action,
                "keywords": keywords
            })

        self.logger.info(
            "Loaded %d active DB rules (%d recruiter, %d anti-recruiter)",
            len(self.rules),
            len(self.recruiter_keywords),
            len(self.anti_recruiter_keywords)
        )

    # --------------------------------------------------
    # ML
    # --------------------------------------------------
    def _load_ml_model(self):
        try:
            model_dir = self.config.get('filters', {}).get('ml_model_dir', '../models')
            classifier_path = os.path.join(model_dir, 'classifier.pkl')
            vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')

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
    # UTILITIES
    # --------------------------------------------------
    def _extract_email(self, from_header: str) -> str | None:
        _, email = parseaddr(from_header or "")
        email = email.lower().strip()
        return email if "@" in email else None

    def _looks_like_random_token(self, token: str) -> bool:
        if len(token) >= 10 and re.search(r'\d', token):
            return True
        if re.fullmatch(r'[a-f0-9]{8,}', token):
            return True
        return False

    def _rule_matches(self, email: str, rule: Dict) -> bool:
        local, domain = email.split("@", 1)

        for kw in rule["keywords"]:
            if rule["match_type"] == "exact":
                if email == kw or domain == kw or local == kw:
                    return True

            elif rule["match_type"] == "contains":
                if kw in email:
                    return True

            elif rule["match_type"] == "regex":
                if kw.search(email):
                    return True

        return False

    # --------------------------------------------------
    # JUNK FILTER (DB FIRST)
    # --------------------------------------------------
    def is_junk_email(self, from_header: str) -> bool:
        email = self._extract_email(from_header)
        if not email:
            return True

        # 1️⃣ DB rules (priority order)
        for rule in self.rules:
            if self._rule_matches(email, rule):
                self.logger.debug(
                    "Email %s %s by rule [%s]",
                    email,
                    rule["action"],
                    rule["category"]
                )
                return rule["action"] == "block"

        # 2️⃣ Behavioral fallback
        local = email.split("@")[0]
        tokens = re.split(r"[._\-+]", local)

        for token in tokens:
            if self._looks_like_random_token(token):
                return True

        return False

    # --------------------------------------------------
    # RECRUITER CLASSIFICATION
    # --------------------------------------------------
    def is_recruiter_email(self, subject: str, body: str, from_header: str) -> bool:
        if self.is_junk_email(from_header):
            return False

        # ML classifier
        if self.use_ml and self.classifier and self.vectorizer:
            try:
                features = self.vectorizer.transform(
                    [f"{subject} {body} {from_header}"]
                )
                return self.classifier.predict(features)[0] == 1
            except Exception as e:
                self.logger.error(f"ML classification error: {e}")
                return False

        text = f"{subject} {body}".lower()
        subject_lower = subject.lower()
        body_lower = body.lower()

        # Strong negative signals
        anti_hits = sum(
            1 for kw in self.anti_recruiter_keywords if kw in text
        )
        if anti_hits >= 4:
            return False

        subject_hits = sum(
            1 for kw in self.recruiter_keywords if kw in subject_lower
        )
        body_hits = sum(
            1 for kw in self.recruiter_keywords if kw in body_lower
        )

        return (
            subject_hits >= 1 or
            body_hits >= 2 or
            (subject_hits + body_hits) >= 1
        )

    # --------------------------------------------------
    # CALENDAR
    # --------------------------------------------------
    def is_calendar_invite(self, email_message) -> bool:
        try:
            for part in email_message.walk():
                if part.get_content_type() == "text/calendar":
                    return True
            return False
        except Exception:
            return False

    # --------------------------------------------------
    # PIPELINE
    # --------------------------------------------------
    def filter_emails(self, emails: List[Dict], cleaner) -> List[Dict]:
        filtered = []

        for email_data in emails:
            try:
                msg = email_data['message']
                from_header = msg.get('From', '')
                subject = msg.get('Subject', '')

                # Always include calendar invites
                if self.config.get('processing', {}) \
                        .get('calendar_invites', {}) \
                        .get('process', True):
                    if self.is_calendar_invite(msg):
                        filtered.append(email_data)
                        continue

                # Junk filter
                if self.is_junk_email(from_header):
                    continue

                body = cleaner.extract_body(msg)

                if self.is_recruiter_email(subject, body, from_header):
                    email_data['clean_body'] = body
                    filtered.append(email_data)

            except Exception as e:
                self.logger.error(f"Error filtering email: {e}")

        self.logger.info(
            "Filtered %d emails from %d total",
            len(filtered),
            len(emails)
        )
        return filtered
