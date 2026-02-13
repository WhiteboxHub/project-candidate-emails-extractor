import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractor.core.database import get_db_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_data():
    print(f"Connecting to database at {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}...")
    db = get_db_client()
    
    try:
        # Check recent vendor contacts
        query = """
        SELECT id, email, first_name, company, job_position, extraction_date, job_source 
        FROM vendor_contact_extracts 
        ORDER BY id DESC 
        LIMIT 5
        """
        results = db.execute_query(query)
        
        print("\n" + "="*80)
        print("LATEST 5 VENDOR CONTACTS IN LOCAL DB")
        print("="*80)
        
        if not results:
            print("No records found.")
        else:
            for row in results:
                print(f"ID: {row.get('id')}")
                print(f"Email: {row.get('email')}")
                print(f"Name: {row.get('first_name')}")
                print(f"Company: {row.get('company')}")
                print(f"Position: {row.get('job_position')}")
                print(f"Source: {row.get('job_source')}")
                print(f"Date: {row.get('extraction_date')}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error querying database: {e}")

if __name__ == "__main__":
    check_data()
