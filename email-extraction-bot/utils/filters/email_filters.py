import re
import os
import joblib
import logging
from typing import Dict, List
from email.utils import parseaddr

logger = logging.getLogger(__name__)


class EmailFilter:
    """
    DB-driven email filtering engine.
    - Priority-based allow/block rules
    - Content scoring using DB weights & targets
    - Optional ML classifier
    - Calendar invite support
    """

    HUMAN_LOCAL_PATTERN = re.compile(
        r"^[a-z]+([._-][a-z]+){0,2}$", re.IGNORECASE)

    def __init__(self, config: dict, db_conn):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # DB-loaded rules (priority ordered)
        self.rules: List[Dict] = []

        # Content rules
        self.content_rules: List[Dict] = []

        # ML classifier
        self.use_ml = config.get("filters", {}).get("use_ml_classifier", False)
        self.classifier = None
        self.vectorizer = None

        # Load rules from DB
        self._load_rules_from_db(db_conn)

        if self.use_ml:
            self._load_ml_model()

    # --------------------------------------------------
    # DB RULE LOADING
    # --------------------------------------------------
    def _load_rules_from_db(self, conn):
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT category, keywords, match_type, action, priority, weight, target
            FROM job_automation_keywords
            WHERE is_active = 1
            ORDER BY priority ASC
            """
        )
        rows = cursor.fetchall()
        cursor.close()

        for row in rows:
            keywords = [
                k.strip().lower() for k in row["keywords"].split(",") if k.strip()
            ]

            # Compile regex if needed
            if row["match_type"] == "regex":
                keywords = [re.compile(k, re.IGNORECASE) for k in keywords]

            rule = {
                "category": row["category"],
                "match_type": row["match_type"],
                "action": row["action"],
                "priority": row["priority"],
                "keywords": keywords,
                "weight": row.get("weight", 1),
                "target": row.get("target", "both"),
            }

            # Separate content rules
            if row["category"] in ["recruiter_keywords", "anti_recruiter_keywords"]:
                self.content_rules.append(rule)
            else:
                self.rules.append(rule)

        self.logger.info(
            "Loaded %d DB rules (%d content rules)",
            len(self.rules),
            len(self.content_rules),
        )

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
    def _extract_email(self, from_header: str) -> str | None:
        _, email = parseaddr(from_header or "")
        email = email.lower().strip()
        return email if "@" in email else None

    def _looks_like_random_token(self, token: str) -> bool:
        if len(token) >= 10 and re.search(r"\d", token):
            return True
        if re.fullmatch(r"[a-f0-9]{8,}", token):
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
    # SENDER FILTER
    # --------------------------------------------------
    def is_junk_email(self, from_header: str) -> bool:
        email = self._extract_email(from_header)
        if not email:
            return True

        for rule in self.rules:
            if self._rule_matches(email, rule):
                return rule["action"] == "block"

        # Fallback for random-looking local part
        local = email.split("@")[0]
        tokens = re.split(r"[._\-+]", local)
        for token in tokens:
            if self._looks_like_random_token(token):
                return True

        return False

    def check_sender(self, email_data: Dict) -> str:
        msg = email_data["message"]
        from_header = msg.get("From", "")
        email = self._extract_email(from_header)
        if not email:
            return "block"

        for rule in self.rules:
            if self._rule_matches(email, rule):
                self.logger.debug(
                    "Sender %s → %s (%s)", email, rule["action"], rule["category"]
                )
                return rule["action"]

        local = email.split("@")[0]
        tokens = re.split(r"[._\-+]", local)
        for token in tokens:
            if self._looks_like_random_token(token):
                return "block"
        return "allow"

    # --------------------------------------------------
    # CONTENT SCORING
    # --------------------------------------------------
    def score_content(self, subject: str, body: str) -> int:
        score = 0
        subject_lower = subject.lower()
        body_lower = body.lower()
        full_text = f"{subject_lower} {body_lower}"

        for rule in self.content_rules:
            weight = rule["weight"]
            target = rule["target"]

            for kw in rule["keywords"]:
                hit = False
                if target in ["subject", "both"] and kw in subject_lower:
                    hit = True
                if target in ["body", "both"] and kw in body_lower:
                    hit = True
                if hit:
                    score += weight

        return score

    # --------------------------------------------------
    # RECRUITER DETECTION
    # --------------------------------------------------
    def is_recruiter_email(self, subject: str, body: str, from_header: str) -> bool:
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
        try:
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
