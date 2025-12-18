"""
UID Tracker - Manages last processed email UID per candidate
Prevents re-processing same emails on subsequent runs
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class UIDTracker:
    """Track last processed UID per email account"""
    
    def __init__(self, tracker_file: str = 'last_run.json'):
        """
        Initialize UID tracker
        
        Args:
            tracker_file: Path to JSON file storing last run data
        """
        self.tracker_file = Path(tracker_file)
        self.data = self._load()
        self.logger = logging.getLogger(__name__)
    
    def _load(self) -> Dict:
        """Load last run data from JSON file"""
        if not self.tracker_file.exists():
            logger.info(f"No last_run.json found - starting fresh (will process all emails)")
            return {}
        
        try:
            with open(self.tracker_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded last_run.json with {len(data)} accounts")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.tracker_file}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {self.tracker_file}: {e}")
            return {}
    
    def _save(self):
        """Save last run data to JSON file"""
        try:
            # Create directory if doesn't exist
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write with pretty formatting
            with open(self.tracker_file, 'w') as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            
            logger.debug(f"Saved last_run.json with {len(self.data)} accounts")
        except Exception as e:
            logger.error(f"Error saving {self.tracker_file}: {e}")
    
    def get_last_uid(self, email: str) -> Optional[str]:
        """
        Get last processed UID for an email account
        
        Args:
            email: Email address (lowercase)
            
        Returns:
            Last UID string or None if first run
        """
        email = email.lower()
        
        if email not in self.data:
            logger.info(f"First run for {email} - will process all emails")
            return None
        
        last_uid = self.data[email].get('last_uid')
        last_run = self.data[email].get('last_run', 'unknown')
        
        logger.info(f"Last run for {email}: UID {last_uid} on {last_run}")
        return last_uid
    
    def update_last_uid(self, email: str, uid: str):
        """
        Update last processed UID for an email account
        
        Args:
            email: Email address (lowercase)
            uid: Last processed UID
        """
        email = email.lower()
        
        # Update data
        self.data[email] = {
            'last_uid': str(uid),
            'last_run': datetime.now().isoformat()
        }
        
        # Save to file
        self._save()
        
        logger.info(f"Updated {email}: last_uid={uid}")
    
    def get_all_tracked_accounts(self) -> list:
        """Get list of all tracked email accounts"""
        return list(self.data.keys())
    
    def remove_account(self, email: str):
        """
        Remove tracking for an account (forces full re-process next run)
        
        Args:
            email: Email address to remove
        """
        email = email.lower()
        
        if email in self.data:
            del self.data[email]
            self._save()
            logger.info(f"Removed tracking for {email}")
    
    def get_stats(self) -> Dict:
        """Get statistics about tracked accounts"""
        if not self.data:
            return {
                'total_accounts': 0,
                'oldest_run': None,
                'newest_run': None
            }
        
        run_dates = []
        for account_data in self.data.values():
            last_run = account_data.get('last_run')
            if last_run:
                try:
                    run_dates.append(datetime.fromisoformat(last_run))
                except:
                    pass
        
        return {
            'total_accounts': len(self.data),
            'oldest_run': min(run_dates).isoformat() if run_dates else None,
            'newest_run': max(run_dates).isoformat() if run_dates else None
        }
    
    def reset_all(self):
        """Reset all tracking (forces full re-process for all accounts)"""
        self.data = {}
        self._save()
        logger.warning("Reset all UID tracking - will process all emails on next run")


# Singleton instance
_tracker_instance = None

def get_uid_tracker(tracker_file: str = 'last_run.json') -> UIDTracker:
    """Get global UID tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = UIDTracker(tracker_file)
    return _tracker_instance
