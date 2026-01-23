import imaplib
import logging
from typing import Optional, List

class GmailIMAPConnector:
    """
    Gmail IMAP connector for email fetching
    Hardcoded to use imap.gmail.com:993
    """
    
    # Hardcoded IMAP settings (same for all candidates)
    IMAP_SERVER = 'imap.gmail.com'
    IMAP_PORT = 993
    
    def __init__(self, email: str, password: str):
        """
        Initialize IMAP connector
        
        Args:
            email: Email address
            password: IMAP app password
        """
        self.email = email
        self.password = password
        self.connection = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Establish IMAP connection
        
        Returns:
            True if connected, False otherwise
        """
        try:
            self.logger.info(f"Connecting to {self.IMAP_SERVER}:{self.IMAP_PORT}...")
            self.connection = imaplib.IMAP4_SSL(self.IMAP_SERVER, self.IMAP_PORT)
            self.connection.login(self.email, self.password)
            self.logger.info(f"Successfully connected to {self.email}")
            return True
            
        except imaplib.IMAP4.error as e:
            self.logger.error(f"IMAP authentication failed for {self.email}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to IMAP for {self.email}: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if IMAP connection is active
        
        Returns:
            True if connected, False otherwise
        """
        try:
            if self.connection is None:
                return False
            # Try to check connection status
            status = self.connection.noop()
            return status[0] == 'OK'
        except:
            return False
    
    def disconnect(self):
        """Close IMAP connection"""
        try:
            if self.connection:
                self.connection.logout()
                self.logger.info(f"Disconnected from {self.email}")
        except Exception as e:
            self.logger.error(f"Error disconnecting: {str(e)}")
    
    def select_folder(self, folder: str = "INBOX") -> bool:
        """
        Select email folder
        
        Args:
            folder: Folder name (default: INBOX)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            status, messages = self.connection.select(folder)
            if status == 'OK':
                num_messages = int(messages[0])
                self.logger.info(f"Selected {folder} - {num_messages} emails")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error selecting folder {folder}: {str(e)}")
            return False
    
    def get_email_uids(self, search_criteria: str = "ALL") -> List[bytes]:
        """
        Get list of email UIDs
        
        Args:
            search_criteria: IMAP search criteria (default: ALL)
            
        Returns:
            List of email UIDs
        """
        try:
            status, messages = self.connection.uid('search', None, search_criteria)
            if status == 'OK':
                uids = messages[0].split()
                self.logger.info(f"Found {len(uids)} emails matching criteria: {search_criteria}")
                return uids
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching UIDs: {str(e)}")
            return []
    
    def fetch_email(self, uid: bytes):
        """
        Fetch a single email by UID
        
        Args:
            uid: Email UID
            
        Returns:
            Email message or None
        """
        try:
            status, data = self.connection.uid('fetch', uid, '(RFC822)')
            if status == 'OK':
                return data[0][1]
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching email UID {uid}: {str(e)}")
            return None
