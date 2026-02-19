import logging
from typing import Optional

logger = logging.getLogger(__name__)

class BERTPreprocessor:
    """
    Standardizes raw job listing data into a structured format for BERT classification.
    """
    
    def __init__(self, max_desc_len: int = 1500):
        self.max_desc_len = max_desc_len

    def format_input(
        self, 
        title: Optional[str] = "", 
        company: Optional[str] = "", 
        location: Optional[str] = "", 
        description: Optional[str] = ""
    ) -> str:
        """
        Combines job fields into a single string.
        Format: [TITLE] {title} [COMPANY] {company} [LOCATION] {location} [CONTEXT] {description}
        """
        # Clean inputs
        t = (title or "N/A").strip()
        c = (company or "N/A").strip()
        l = (location or "N/A").strip()
        d = (description or "").strip()
        
        # Truncate description to avoid exceeding BERT token limits
        if len(d) > self.max_desc_len:
            d = d[:self.max_desc_len] + "..."

        # Assemble structured text
        # Using special tags to help BERT understand field boundaries
        rich_text = (
            f"[TITLE] {t} "
            f"[COMPANY] {c} "
            f"[LOCATION] {l} "
            f"[CONTEXT] {d}"
        )
        
        return rich_text
