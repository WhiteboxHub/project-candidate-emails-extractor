from typing import Set, Optional
import logging

logger = logging.getLogger(__name__)

class DeduplicationCache:
    """
    Manages deduplication state for the extraction workflow.
    
    Responsibilities:
    1. Intra-run deduplication: Tracks emails seen in the current run to avoid processing/saving the same contact twice.
    2. Database cache: Optionally caches known existing emails from the database to reduce query load.
    """
    
    def __init__(self):
        self.seen_emails_run: Set[str] = set()
        self.known_db_emails: Set[str] = set()
        self._db_cache_populated = False

    def is_seen_in_run(self, email: str) -> bool:
        """Check if an email has already been processed in the current run."""
        if not email:
            return False
        return email.strip().lower() in self.seen_emails_run

    def is_known_in_db(self, email: str) -> bool:
        """Check if an email is known to exist in the database (if cache populated)."""
        if not email or not self._db_cache_populated:
            return False
        return email.strip().lower() in self.known_db_emails

    def mark_seen_in_run(self, email: str):
        """Mark an email as processed in the current run."""
        if email:
            self.seen_emails_run.add(email.strip().lower())

    def add_known_db_emails(self, emails: Set[str]):
        """Add a batch of known emails from the database to the cache."""
        normalized = {e.strip().lower() for e in emails if e}
        self.known_db_emails.update(normalized)
        self._db_cache_populated = True
        logger.debug(f"Added {len(normalized)} emails to DB deduplication cache. Total: {len(self.known_db_emails)}")

    def clear_run_cache(self):
        """Clear the intra-run cache (e.g., between distinct workflows if reused)."""
        self.seen_emails_run.clear()
        
    def get_stats(self) -> dict:
        return {
            "seen_in_run": len(self.seen_emails_run),
            "known_in_db": len(self.known_db_emails)
        }
