import re
from bs4 import BeautifulSoup
import logging
import html as html_lib

logger = logging.getLogger(__name__)


class EmailCleaner:
    """Clean and sanitize email content for extraction"""

    # Max body length we'll process (characters)
    _MAX_BODY = 8000
    # Always keep the last N chars regardless of truncation (signature zone)
    _SIG_ZONE = 700

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def clean_html(self, html_content: str) -> str:
        """Remove HTML tags and extract clean text from an HTML payload."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text with newlines preserved
            text = soup.get_text(separator="\n")
            text = self._remove_quoted_replies(text)
            text = self._normalize_whitespace(text)
            return text.strip()

        except Exception as e:
            self.logger.error(f"Error cleaning HTML: {str(e)}")
            return html_content

    def extract_body(self, email_message) -> str:
        """
        Extract email body from message object.

        Strategy
        --------
        1. Prefer plain text over HTML.
        2. Clean residual HTML entities (&nbsp; etc.) from plain-text payloads.
        3. Truncate very long bodies to _MAX_BODY chars BUT always preserve the
           last _SIG_ZONE chars (the signature zone) so name/title/company are
           never lost.

        Returns
        -------
        Cleaned body string (empty string on failure).
        """
        body = ""
        try:
            if email_message.is_multipart():
                text_body = None
                html_body = None

                for part in email_message.walk():
                    content_type = part.get_content_type()

                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text_body = payload.decode("utf-8", errors="ignore")

                    elif content_type == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            html_body = payload.decode("utf-8", errors="ignore")

                if text_body:
                    body = self._clean_html_entities(text_body)
                elif html_body:
                    body = self.clean_html(html_body)

            else:
                payload = email_message.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")
                    if "<html" in body.lower():
                        body = self.clean_html(body)
                    else:
                        body = self._clean_html_entities(body)

            # Final cleaning
            if body:
                body = self._remove_quoted_replies(body)
                body = self._normalize_whitespace(body)
                body = self._smart_truncate(body)

            return body.strip()

        except Exception as e:
            self.logger.error(f"Error extracting email body: {str(e)}")
            return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_quoted_replies(self, text: str) -> str:
        """Remove quoted email replies and forwarded message headers."""
        patterns = [
            r"On .+ wrote:",
            r"From:.+Sent:.+To:.+Subject:",
            r"_{5,}",
            r"-{5,}",
            r"Begin forwarded message:",
        ]
        for pattern in patterns:
            parts = re.split(pattern, text, maxsplit=1)
            text = parts[0]
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/blank lines."""
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines)

    def _clean_html_entities(self, text: str) -> str:
        """Unescape HTML entities that survive in plain-text payloads."""
        text = html_lib.unescape(text)
        # Strip any remaining bare named entities we missed
        text = re.sub(r"&[a-zA-Z]{2,8};", " ", text)
        return text

    def _smart_truncate(self, text: str) -> str:
        """
        Truncate to _MAX_BODY chars while always preserving the last
        _SIG_ZONE chars (recruiter signature block).
        """
        if len(text) <= self._MAX_BODY:
            return text

        head = text[: self._MAX_BODY - self._SIG_ZONE]
        tail = text[-self._SIG_ZONE :]
        return head + "\n\n" + tail
