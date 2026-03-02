import re
from typing import Dict, List
import logging
from ..filtering.repository import get_filter_repository
from ..filtering.ml_filter import MLFilter

logger = logging.getLogger(__name__)


class EmailFilter:
    """Filter and classify emails — recruiter vs. marketing/junk."""

    # How many non-recruiter body signals trigger pre-extraction discard
    _NON_RECRUITER_THRESHOLD = 3

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.filter_repo = get_filter_repository()
        keyword_lists = self.filter_repo.get_keyword_lists()

        # Primary recruiter/anti-recruiter signals (already in CSV)
        self.recruiter_keywords = keyword_lists.get("recruiter_keywords", [])
        self.anti_recruiter_keywords = keyword_lists.get("anti_recruiter_keywords", [])

        # NEW: pre-extraction body signals from updated keywords.csv
        self.non_recruiter_body_signals = keyword_lists.get("non_recruiter_body_signals", [])
        self.recruiter_email_signals = keyword_lists.get("recruiter_email_signals", [])

        # Allowed staffing domains — these are auto-classified recruiter
        raw_staffing = keyword_lists.get("allowed_staffing_domain", [])
        self.allowed_staffing_domains = {d.strip().lower() for d in raw_staffing}

        self.use_ml = config.get("filters", {}).get("use_ml_classifier", False)
        self.ml_filter = None
        if self.use_ml:
            self._load_ml_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_ml_model(self):
        model_dir = self.config.get("filters", {}).get("ml_model_dir", "../models")
        ml_filter = MLFilter(model_dir=model_dir)
        if ml_filter.load():
            self.ml_filter = ml_filter
            self.logger.info("ML filtering enabled")
        else:
            self.logger.warning("ML filtering disabled — model files missing")
            self.ml_filter = None
            self.use_ml = False

    def _classify_with_ml(self, subject: str, body: str, from_email: str):
        if not self.ml_filter:
            return None
        return self.ml_filter.predict_recruiter(subject=subject, body=body, from_email=from_email)

    def _classify_with_rules(self, subject: str, body: str, from_email: str) -> bool:
        """
        Multi-signal rule classifier.

        Priority order:
        1. Allowed staffing domain  → always recruiter
        2. Strong recruiter body signals → recruiter
        3. Marketing/spam body signals  → not recruiter
        4. Keyword scoring
        """
        subject_lower = (subject or "").lower()
        body_lower = (body or "").lower()
        combined = f"{subject_lower} {body_lower}"

        # 1. Known staffing / recruiter domain → immediate pass
        if from_email:
            domain = from_email.split("@")[-1].lower().strip()
            if any(domain.endswith(sd) or domain == sd for sd in self.allowed_staffing_domains):
                self.logger.debug(f"Auto-recruiter: staffing domain {domain}")
                return True

        # 2. Strong recruiter email signal in body → pass
        if self.recruiter_email_signals:
            for signal in self.recruiter_email_signals:
                if signal in body_lower:
                    self.logger.debug(f"Auto-recruiter: signal '{signal}' found in body")
                    return True

        # 3. Marketing/spam body signals → fail fast (already checked in
        #    is_marketing_body but checked here again for _classify_with_rules path)
        if self.non_recruiter_body_signals:
            hit_count = sum(1 for s in self.non_recruiter_body_signals if s in combined)
            if hit_count >= self._NON_RECRUITER_THRESHOLD:
                self.logger.debug(f"Rejected: {hit_count} non-recruiter body signals")
                return False

        # 4. Anti-recruiter keyword density
        anti_count = sum(1 for kw in self.anti_recruiter_keywords if kw in combined)
        if anti_count >= 4:
            return False

        # 5. Positive recruiter keyword scoring
        subject_score = sum(1 for kw in self.recruiter_keywords if kw in subject_lower)
        body_score = sum(1 for kw in self.recruiter_keywords if kw in body_lower)

        if subject_score >= 1:
            return True
        if body_score >= 2:
            return True
        if subject_score + body_score >= 1:
            return True

        return False

    def _extract_clean_email(self, from_header: str) -> str:
        """Extract bare email address from a From header."""
        if not from_header:
            return ""
        match = re.search(
            r"(?:<|\(|^)([\w.\-+]+@[\w.\-]+)(?:>|\)|$)",
            from_header,
            re.IGNORECASE,
        )
        return match.group(1).lower() if match else ""

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def is_junk_email(self, from_header: str) -> bool:
        """Check if email is junk/automated/system using database filters."""
        email = self._extract_clean_email(from_header)
        if not email or "@" not in email:
            return True
        action = self.filter_repo.check_email(email)
        if action == "block":
            self.logger.debug(f"Blocked by filter: {email}")
            return True
        return False

    def is_marketing_body(self, body: str) -> bool:
        """
        NEW: Pre-extraction body gate.
        Returns True (= discard) when ≥ _NON_RECRUITER_THRESHOLD marketing
        signals are found in the body.  Saves expensive NLP time.
        """
        if not self.non_recruiter_body_signals or not body:
            return False
        body_lower = body.lower()
        hit_count = sum(1 for s in self.non_recruiter_body_signals if s in body_lower)
        if hit_count >= self._NON_RECRUITER_THRESHOLD:
            self.logger.debug(f"Pre-NLP discard: {hit_count} marketing signals in body")
            return True
        return False

    def is_recruiter_email(self, subject: str, body: str, from_email: str) -> bool:
        """Classify if email is from a recruiter."""
        if self.is_junk_email(from_email):
            return False

        # Pre-extraction marketing body gate
        if self.is_marketing_body(body):
            return False

        if self.use_ml:
            ml_result = self._classify_with_ml(subject, body, from_email)
            if ml_result is not None:
                return ml_result

        clean_from = self._extract_clean_email(from_email) or from_email
        return self._classify_with_rules(subject, body, clean_from)

    def is_calendar_invite(self, email_message) -> bool:
        """Check if email is a calendar invite."""
        try:
            for part in email_message.walk():
                if part.get_content_type() == "text/calendar":
                    return True
            return False
        except Exception:
            return False

    def filter_emails(self, emails: List[Dict], cleaner) -> tuple:
        """
        Filter email list to keep only recruiter / calendar emails.

        Returns
        -------
        (filtered_emails, filter_stats)
        """
        filtered = []
        junk_count = 0
        not_recruiter_count = 0
        calendar_count = 0

        for email_data in emails:
            try:
                msg = email_data["message"]
                from_header = msg.get("From", "")
                subject = msg.get("Subject", "")

                # Always include calendar invites
                if self.config.get("processing", {}).get("calendar_invites", {}).get("process", True):
                    if self.is_calendar_invite(msg):
                        self.logger.debug(f"Including calendar invite from {from_header}")
                        calendar_count += 1
                        filtered.append(email_data)
                        continue

                # Skip junk emails (domain/prefix blocklist)
                if self.is_junk_email(from_header):
                    junk_count += 1
                    continue

                # Extract and clean body
                body = cleaner.extract_body(msg)

                # Pre-extraction marketing body gate (saves NLP time)
                if self.is_marketing_body(body):
                    not_recruiter_count += 1
                    continue

                # Full recruiter classification
                if self.is_recruiter_email(subject, body, from_header):
                    email_data["clean_body"] = body
                    filtered.append(email_data)
                else:
                    not_recruiter_count += 1

            except Exception as e:
                self.logger.error(f"Error filtering email: {str(e)}")
                continue

        filter_stats = {
            "total": len(emails),
            "passed": len(filtered),
            "junk": junk_count,
            "not_recruiter": not_recruiter_count,
            "calendar_invites": calendar_count,
        }
        self.logger.info(
            f"Filtered {len(filtered)} emails from {len(emails)} total "
            f"(Junk: {junk_count}, Not recruiter: {not_recruiter_count}, "
            f"Calendar: {calendar_count})"
        )
        return filtered, filter_stats
