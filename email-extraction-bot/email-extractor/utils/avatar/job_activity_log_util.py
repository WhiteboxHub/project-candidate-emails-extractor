import mysql.connector
import logging
from typing import Optional

class JobActivityLogUtil:
    """
    Utility for logging job activity to job_activity_log table
    Tracks vendor contacts extracted per candidate per day
    """
    
    def __init__(self, db_config: dict, employee_id: int, job_id: int = 25):
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)
        self.job_id = job_id  # Hardcoded: 25 = 'Bot Candidate Email Extractor'
        self.employee_id = employee_id  # Configurable employee ID
    

    
    def log_activity(self, candidate_id: int, contacts_extracted: int):
        """
        Log job activity for a candidate (inserts new row each time)
        
        Args:
            candidate_id: candidate.id (FK)
            contacts_extracted: Number of vendor contacts saved to database
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Insert new row for each run
            cursor.execute("""
                INSERT INTO job_activity_log 
                (job_id, candidate_id, employee_id, activity_date, activity_count)
                VALUES (%s, %s, %s, CURDATE(), %s)
            """, (self.job_id, candidate_id, self.employee_id, contacts_extracted))
            
            conn.commit()
            
            self.logger.info(
                f"Activity logged for candidate_id {candidate_id}: "
                f"{contacts_extracted} contacts saved"
            )
            
            cursor.close()
            conn.close()
            
        except mysql.connector.Error as err:
            self.logger.error(f"Database error logging activity: {err}")
        except Exception as e:
            self.logger.error(f"Error logging activity for candidate {candidate_id}: {str(e)}")
    
    def get_today_summary(self) -> dict:
        """
        Get summary of today's extraction activity
        
        Returns:
            Dictionary with summary statistics
        """
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT candidate_id) as candidates_processed,
                    SUM(activity_count) as total_contacts_extracted
                FROM job_activity_log
                WHERE job_id = %s 
                  AND activity_date = CURDATE()
            """, (self.job_id,))
            
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return result or {}
            
        except mysql.connector.Error as err:
            self.logger.error(f"Database error fetching summary: {err}")
            return {}
