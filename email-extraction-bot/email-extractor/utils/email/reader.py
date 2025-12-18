import email
from email.header import decode_header
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EmailReader:
    """Read and fetch emails from IMAP connection"""
    
    def __init__(self, connector):
        self.connector = connector
        self.logger = logging.getLogger(f"{__name__}.{connector.email}")
    
    def fetch_emails(
        self, 
        since_uid: Optional[str] = None, 
        batch_size: int = 100, 
        start_index: int = 0
    ) -> Tuple[List[Dict], Optional[int]]:
        """
        Fetch emails in batches using UID
        
        Args:
            since_uid: Last processed UID (fetch newer emails)
            batch_size: Number of emails per batch
            start_index: Starting index for batch
            
        Returns:
            Tuple of (email_list, next_start_index)
        """
        if not self.connector.is_connected():
            if not self.connector.connect():
                return [], None
        
        # Select INBOX folder before searching
        if not self.connector.select_folder('INBOX'):
            self.logger.error("Failed to select INBOX folder")
            return [], None
        
        try:
            # Build search criteria
            if since_uid:
                try:
                    next_uid = int(since_uid) + 1
                except (ValueError, TypeError):
                    next_uid = since_uid
                criteria = f'(UID {next_uid}:*)'
            else:
                criteria = 'ALL'
            
            # Search for emails
            status, messages = self.connector.connection.uid('search', None, criteria)
            
            if status != 'OK':
                self.logger.error(f"Email search failed: {status}")
                return [], None
            
            email_uids = messages[0].split()
            total_emails = len(email_uids)
            
            if total_emails == 0 or start_index >= total_emails:
                self.logger.info(f"No new emails to process")
                return [], None
            
            # Calculate batch slice
            end_index = min(start_index + batch_size, total_emails)
            batch_uids = email_uids[-end_index:-start_index] if start_index > 0 else email_uids[-end_index:]
            batch_uids = list(reversed(batch_uids))  # Newest first
            
            self.logger.info(f"Fetching {len(batch_uids)} emails (batch {start_index}-{end_index}/{total_emails})")
            
            # Fetch emails
            emails = []
            for uid in batch_uids:
                try:
                    email_data = self._fetch_single_email(uid)
                    if email_data:
                        emails.append(email_data)
                except Exception as e:
                    self.logger.error(f"Error fetching email UID {uid}: {str(e)}")
                    continue
            
            next_start_index = end_index if end_index < total_emails else None
            return emails, next_start_index
            
        except Exception as e:
            self.logger.error(f"Error in fetch_emails: {str(e)}")
            return [], None
    
    def _fetch_single_email(self, uid) -> Optional[Dict]:
        """Fetch a single email by UID with better parsing"""
        try:
            status, msg_data = self.connector.connection.uid('fetch', uid, '(RFC822)')
            
            if status != 'OK' or not msg_data or not msg_data[0]:
                return None
            
            raw_email = msg_data[0][1]
            if not raw_email:
                return None
            
            email_message = email.message_from_bytes(raw_email)
            
            # Validate email has minimum required fields
            if not email_message.get('From'):
                self.logger.debug(f"Email UID {uid} has no From header - skipping")
                return None
            
            return {
                'uid': uid.decode() if isinstance(uid, bytes) else str(uid),
                'message': email_message,
                'raw': raw_email,
                'subject': self.clean_text(email_message.get('Subject', '')),
                'from': email_message.get('From', ''),
                'to': email_message.get('To', ''),
                'date': email_message.get('Date', '')
            }
        except Exception as e:
            self.logger.error(f"Error fetching email UID {uid}: {str(e)}")
            return None
    
    @staticmethod
    def clean_text(text):
        """Decode email header text"""
        if text is None:
            return ""
        try:
            decoded_text = decode_header(text)[0][0]
            if isinstance(decoded_text, bytes):
                return decoded_text.decode('utf-8', errors='ignore')
            return str(decoded_text)
        except:
            return str(text)
