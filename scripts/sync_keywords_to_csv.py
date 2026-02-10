import csv
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Set
from utils.api_client import get_api_client
import shutil
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CSV configuration
CSV_FILE = Path(__file__).parent / "keywords.csv"
CSV_COLUMNS = [
    'id', 'category', 'source', 'keywords', 'match_type',
    'action', 'priority', 'context', 'is_active',
    'created_at', 'updated_at'
]


def fetch_keywords_from_database() -> List[Dict]:
    """Fetch all keywords from the database via API."""
    api_url = os.getenv("API_BASE_URL")
    api_email = os.getenv("API_EMAIL")

    if not api_url or not api_email:
        logger.error("API credentials missing in environment variables.")
        return []

    try:
        logger.info(f"Fetching keywords from API: {api_url}")
        client = get_api_client()
        response = client.get("/api/job-automation-keywords")
        
        # Handle different response formats like in candidate_util.py
        if isinstance(response, list):
            keywords = response
        elif isinstance(response, dict):
            # Try common keys for list data
            keywords = response.get("data", response.get("items", response.get("keywords", [])))
            
            # Log response structure for debugging
            if not keywords:
                logger.warning(f"No keywords found. Response keys: {list(response.keys())}")
                logger.debug(f"Full response: {response}")
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            keywords = []
        
        logger.info(f"Fetched {len(keywords)} keywords from database")
        if keywords and len(keywords) > 0:
            logger.info(f"Sample keyword category: {keywords[0].get('category', 'N/A')}")
        
        return keywords
    except Exception as e:
        logger.error(f"Failed to fetch keywords: {e}", exc_info=True)
        return []


def load_existing_csv() -> (List[Dict], Set[int]):
    """Load existing CSV data if it exists."""
    if not CSV_FILE.exists():
        logger.info(f"CSV file not found: {CSV_FILE}")
        return [], set()

    rows, ids = [], set()
    try:
        with open(CSV_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                try:
                    ids.add(int(row['id']))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid ID in CSV: {row.get('id')}")
        logger.info(f"Loaded {len(rows)} rows from CSV")
        return rows, ids
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return [], set()


def format_row(keyword: Dict) -> Dict:
    """Format a database keyword for CSV."""
    is_active = keyword.get("is_active", 1)
    if isinstance(is_active, bool):
        is_active = int(is_active)
    return {
        'id': keyword.get('id', ''),
        'category': keyword.get('category', ''),
        'source': keyword.get('source', 'email_extractor'),
        'keywords': keyword.get('keywords', ''),
        'match_type': keyword.get('match_type', 'contains'),
        'action': keyword.get('action', 'block'),
        'priority': keyword.get('priority', 100),
        'context': keyword.get('context', ''),
        'is_active': is_active,
        'created_at': keyword.get('created_at', ''),
        'updated_at': keyword.get('updated_at', ''),
    }


def backup_csv():
    """Backup the existing CSV file."""
    if CSV_FILE.exists():
        backup_file = CSV_FILE.with_suffix(".csv.backup")
        shutil.copy2(CSV_FILE, backup_file)
        logger.info(f"Created backup: {backup_file}")


def write_csv(rows: List[Dict]):
    """Write keyword rows to CSV."""
    try:
        backup_csv()
        with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Wrote {len(rows)} rows to {CSV_FILE}")
    except Exception as e:
        logger.error(f"Error writing CSV: {e}")
        raise


def sync_keywords(dry_run: bool = False):
    """Sync keywords from database to CSV."""
    logger.info("=" * 80)
    logger.info(f"Starting keyword sync {'(DRY RUN)' if dry_run else ''}")
    logger.info("=" * 80)

    db_keywords = fetch_keywords_from_database()
    if not db_keywords:
        logger.warning("No keywords fetched. Check API configuration.")
        return

    existing_rows, existing_ids = load_existing_csv()
    new_rows, updated_count = [], 0

    id_to_row = {int(r['id']): r for r in existing_rows if r.get('id') and str(r['id']).isdigit()}

    for kw in db_keywords:
        kw_id = kw.get('id')
        formatted = format_row(kw)

        if kw_id not in id_to_row:
            new_rows.append(formatted)
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}New keyword: ID={kw_id}, Category={kw.get('category')}")
        else:
            id_to_row[int(kw_id)] = formatted
            updated_count += 1

    all_rows = list(id_to_row.values()) + new_rows
    all_rows.sort(key=lambda x: int(x['id']) if x.get('id') and str(x['id']).isdigit() else 999999)

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Updated {updated_count} existing rows")
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Added {len(new_rows)} new rows")

    if dry_run:
        logger.info(f"DRY RUN - Would write {len(all_rows)} rows to {CSV_FILE}")
    else:
        write_csv(all_rows)
        logger.info("Sync completed successfully!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Sync keywords from database to CSV")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    try:
        sync_keywords(dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
