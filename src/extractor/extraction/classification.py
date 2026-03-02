from typing import Tuple, Optional
import logging
from ..filtering.repository import get_filter_repository

logger = logging.getLogger(__name__)


class RecruiterClassifier:
    """
    Classifies whether a contact is likely a recruiter / talent acquisition
    professional.

    Signal priority (highest → lowest):
    1. Sender email is from a known staffing/allowed domain  → recruiter
    2. Title matches a strong recruiter indicator            → recruiter
    3. Title matches a negative indicator (tech role)       → NOT recruiter
    4. Title matches moderate / weak indicators
    5. Body scored against recruiter_email_signals  (new)
    6. Body scored against recruiter_keywords / anti_recruiter_keywords
    7. No signal → NOT recruiter
    """

    # Body keyword hit thresholds
    _RECRUITER_BODY_SCORE_PASS = 0.4   # weighted score to call recruiter
    _ANTI_BODY_THRESHOLD = 4           # anti-keyword count to call NOT recruiter

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.filter_repo = get_filter_repository()

        lists = self.filter_repo.get_keyword_lists()

        self.strong_indicators    = self._as_set(lists, "recruiter_title_strong")
        self.moderate_indicators  = self._as_set(lists, "recruiter_title_moderate")
        self.weak_indicators      = self._as_set(lists, "recruiter_title_weak")
        self.negative_indicators  = self._as_set(lists, "recruiter_title_negative")
        self.context_indicators   = self._as_list(lists, "recruiter_context_positive")

        # NEW: richer body signals from updated keywords.csv
        self.recruiter_email_signals      = self._as_list(lists, "recruiter_email_signals")
        self.recruiter_body_keywords      = self._as_list(lists, "recruiter_keywords")
        self.anti_recruiter_body_keywords = self._as_list(lists, "anti_recruiter_keywords")

        # Allowed staffing domains for automatic classification
        raw_staffing = lists.get("allowed_staffing_domain", [])
        self.allowed_staffing_domains = {d.strip().lower() for d in raw_staffing}

        self.logger.info(
            "RecruiterClassifier ready: %d strong / %d moderate / %d negative / "
            "%d recruiter signals / %d body keywords",
            len(self.strong_indicators),
            len(self.moderate_indicators),
            len(self.negative_indicators),
            len(self.recruiter_email_signals),
            len(self.recruiter_body_keywords),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _as_set(self, lists: dict, key: str) -> set:
        try:
            return {kw.lower().strip() for kw in lists.get(key, [])}
        except Exception as e:
            self.logger.error(f"Failed to load {key}: {e}")
            return set()

    def _as_list(self, lists: dict, key: str) -> list:
        try:
            return [kw.lower().strip() for kw in lists.get(key, [])]
        except Exception as e:
            self.logger.error(f"Failed to load {key}: {e}")
            return []

    def _score_body(self, body: str) -> Tuple[float, str]:
        """
        Return (score 0–1, reason) based on body keyword analysis.
        Uses recruiter_email_signals (strong), recruiter_keywords (moderate),
        and anti_recruiter_keywords (negative).
        """
        body_lower = body.lower()

        # Strong recruiter signals
        strong_hits = [s for s in self.recruiter_email_signals if s in body_lower]
        if strong_hits:
            return 1.0, f"Strong body signal: {strong_hits[0]}"

        # Anti-recruiter count
        anti_count = sum(1 for kw in self.anti_recruiter_body_keywords if kw in body_lower)
        if anti_count >= self._ANTI_BODY_THRESHOLD:
            return 0.0, f"Anti-recruiter body ({anti_count} signals)"

        # Positive keyword scoring
        pos_hits = sum(1 for kw in self.recruiter_body_keywords if kw in body_lower)
        # Each hit adds 0.15, capped at 1.0
        score = min(1.0, pos_hits * 0.15)
        if score >= self._RECRUITER_BODY_SCORE_PASS:
            return score, f"Body recruiter keywords: {pos_hits} hits"

        return score, f"Insufficient body signals (score={score:.2f})"

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def is_recruiter(
        self,
        title: Optional[str],
        context: Optional[str] = None,
        sender_email: Optional[str] = None,
    ) -> Tuple[bool, float, str]:
        """
        Determine if the contact is a recruiter.

        Parameters
        ----------
        title         : Job title extracted from signature (may be None)
        context       : Full email body for body-level scoring
        sender_email  : Sender email address for domain-based override

        Returns
        -------
        (is_recruiter: bool, score: float 0–1, reason: str)
        """
        # ── PRIORITY 0: Known staffing domain → always recruiter ──────────────
        if sender_email:
            domain = sender_email.split("@")[-1].lower().strip()
            if any(domain.endswith(sd) or domain == sd for sd in self.allowed_staffing_domains):
                return True, 1.0, f"Allowed staffing domain: {domain}"

        # ── PRIORITY 1: Title-based checks ────────────────────────────────────
        if title:
            title_lower = title.lower()

            # Negative gate (tech roles are not recruiters)
            for ind in self.negative_indicators:
                if ind in title_lower:
                    return False, 0.0, f"Negative title indicator: {ind}"

            # Strong recruiter title → immediate pass
            for ind in self.strong_indicators:
                if ind in title_lower:
                    return True, 1.0, f"Strong title indicator: {ind}"

            # Moderate
            mod_hits = [ind for ind in self.moderate_indicators if ind in title_lower]
            if mod_hits:
                score = 0.6
                # Tiebreak with body if available
                if context:
                    body_score, body_reason = self._score_body(context)
                    combined = min(1.0, score + body_score * 0.3)
                    return combined >= 0.5, combined, f"Moderate title ({mod_hits[0]}) + body {body_reason}"
                return True, score, f"Moderate title indicator: {mod_hits[0]}"

            # Weak
            weak_hits = [ind for ind in self.weak_indicators if ind in title_lower]
            if weak_hits:
                score = 0.3
                if context:
                    body_score, body_reason = self._score_body(context)
                    combined = min(1.0, score + body_score * 0.5)
                    return combined >= 0.5, combined, f"Weak title ({weak_hits[0]}) + body: {body_reason}"
                return False, score, f"Weak title indicator (no body context): {weak_hits[0]}"

        # ── PRIORITY 2: No title → fall back entirely to body scoring ─────────
        if context:
            # Check legacy context indicators first
            context_lower = context.lower()
            for indicator in self.context_indicators:
                if indicator in context_lower:
                    return True, 0.8, f"Context phrase: {indicator}"

            # Full body scoring
            body_score, body_reason = self._score_body(context)
            is_rec = body_score >= self._RECRUITER_BODY_SCORE_PASS
            return is_rec, body_score, f"Body-only scoring: {body_reason}"

        return False, 0.0, "No title or body context available"

    def _analyze_context(self, context: str) -> Tuple[bool, float, str]:
        """Legacy interface — delegates to body scoring."""
        return self.is_recruiter(title=None, context=context)
