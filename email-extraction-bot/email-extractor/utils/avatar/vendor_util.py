import mysql.connector
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VendorUtil:
    """Manage vendor_contact_extracts table operations"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)
    
    def save_contacts(self, contacts: List[Dict]) -> int:
        """
        Save extracted contacts to database with advanced deduplication
        
        Args:
            contacts: List of contact dictionaries
            
        Returns:
            Number of new contacts inserted
        """
        if not contacts:
            self.logger.info("No contacts to save")
            return 0
        
        # Pre-filter contacts for quality
        valid_contacts = []
        for contact in contacts:
            if self._is_valid_contact(contact):
                valid_contacts.append(contact)
            else:
                self.logger.debug(f"Skipped invalid contact: {contact.get('email', 'N/A')}")
        
        if not valid_contacts:
            self.logger.info("No valid contacts after filtering")
            return 0
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            new_count = 0
            
            for contact in valid_contacts:
                email = contact.get('email')
                linkedin = contact.get('linkedin_id')
                name = contact.get('name', '')
                phone = contact.get('phone', '')
                company = contact.get('company', '')
                location = contact.get('location', '')
                source = contact.get('source', '').lower()
                
                # Skip if both email and linkedin are missing
                if not email and not linkedin:
                    continue
                
                # Check for existing record by email
                if email:
                    cursor.execute(
                        "SELECT * FROM vendor_contact_extracts WHERE email = %s",
                        (email,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        # Update if linkedin missing
                        if not row.get("linkedin_id") and linkedin:
                            cursor.execute("""
                                UPDATE vendor_contact_extracts
                                SET linkedin_id=%s, full_name=%s, phone=%s,
                                    company_name=%s, location=%s, source_email=%s
                                WHERE id=%s
                            """, (linkedin, name or row['full_name'], phone or row['phone'],
                                  company or row['company_name'], location or row['location'],
                                  source, row["id"]))
                            conn.commit()
                            self.logger.debug(f"Updated existing contact: {email}")
                        continue
                
                # Check for existing record by linkedin
                if linkedin:
                    cursor.execute(
                        "SELECT * FROM vendor_contact_extracts WHERE linkedin_id = %s",
                        (linkedin,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        # Update if email missing
                        if not row.get("email") and email:
                            cursor.execute("""
                                UPDATE vendor_contact_extracts
                                SET email=%s, full_name=%s, phone=%s,
                                    company_name=%s, location=%s, source_email=%s
                                WHERE id=%s
                            """, (email, name or row['full_name'], phone or row['phone'],
                                  company or row['company_name'], location or row['location'],
                                  source, row["id"]))
                            conn.commit()
                            self.logger.debug(f"Updated existing contact: {linkedin}")
                        continue
                
                # Insert new contact
                cursor.execute("""
                    INSERT INTO vendor_contact_extracts
                    (full_name, source_email, email, phone, linkedin_id, company_name, 
                     location, extraction_date, moved_to_vendor, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURDATE(), 0, NOW())
                """, (name, source, email, phone, linkedin, company, location))
                
                conn.commit()
                new_count += 1
                self.logger.debug(f"Inserted new contact: {email or linkedin}")
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"Saved {new_count} new contacts to database")
            return new_count
            
        except mysql.connector.Error as err:
            self.logger.error(f"Database error saving contacts: {err}")
            return 0
        except Exception as e:
            self.logger.error(f"Error saving contacts: {str(e)}")
            return 0
    
    def _is_valid_contact(self, contact: Dict) -> bool:
        """Validate contact has minimum required quality"""
        try:
            email = contact.get('email', '')
            linkedin = contact.get('linkedin_id', '')
            name = contact.get('name', '')
            
            # Must have email OR linkedin
            if not email and not linkedin:
                return False
            
            # If has email, validate format
            if email:
                if '@' not in email or '.' not in email:
                    return False
                # Skip generic/automated emails
                email_lower = email.lower()
                if any(x in email_lower for x in ['noreply', 'no-reply', 'info@', 'support@', 'admin@']):
                    return False
            
            # If has linkedin, validate it's not a name
            if linkedin:
                if ' ' in linkedin or len(linkedin) > 50:
                    return False
            
            # If has name, basic validation
            if name:
                # Skip single-word names or too long
                words = name.split()
                if len(words) < 2 or len(words) > 4:
                    return False
                # Skip names with numbers
                if any(c.isdigit() for c in name):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating contact: {str(e)}")
            return False
