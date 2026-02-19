#!/usr/bin/env python3
import argparse
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.extractor.connectors.http_api import get_api_client
from src.extractor.preprocessor.bert_preprocessor import BERTPreprocessor
from src.extractor.extraction.llm_classifier import LLMJobClassifier
from src.extractor.persistence.jobs import JobPersistence

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("llm_classification.log")
    ]
)
logger = logging.getLogger("llm_classifier")

class LLMJobClassifyOrchestrator:
    def __init__(self, dry_run: bool = False, batch_size: int = 20, threshold: float = 0.7):
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.audit_log = Path("classification_audit_llm.log")
        
        # Initialize components
        try:
            load_dotenv()
            self.api_client = get_api_client()
            self.persistence = JobPersistence(self.api_client)
            self.preprocessor = BERTPreprocessor()
            self.classifier = LLMJobClassifier(threshold=threshold)
            logger.info("Local LLM components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            sys.exit(1)

    def run(self):
        print("\n" + "="*60)
        print(" STARTING LLM JOB CLASSIFICATION ENGINE")
        print("="*60)
        logger.info(f"Mode:----- {'DRY RUN' if self.dry_run else 'PRODUCTION'} | Batch: {self.batch_size}")
        
        current_skip = 0
        while True:
            # 1. Fetch raw jobs (with pagination skip support)
            raw_jobs, total_fetched = self.persistence.fetch_raw_jobs(limit=self.batch_size, skip=current_skip)
            
            if not raw_jobs:
                if total_fetched > 0:
                    # Batch was full of parsed records, skip them and continue
                    current_skip += total_fetched
                    continue
                    
                # If we were using skip and got nothing at all, maybe try skip=0 once more before quitting
                if current_skip > 0:
                    logger.info("No jobs found with skip. Checking from the beginning...")
                    raw_jobs, total_fetched = self.persistence.fetch_raw_jobs(limit=self.batch_size, skip=0)
                    current_skip = 0
                    if not raw_jobs:
                        break
                else:
                    print("\n" + "-"*60)
                    print(" No new jobs to process. System idling.")
                    print("-"*60 + "\n")
                    break
            
            # If after client-side filtering we have nothing to process, we must skip ahead
            # this happens if the API is ignoring the processing_status filter
            processed_in_batch = 0
            
            print(f"\nProcessing batch of {len(raw_jobs)} candidates (Current Skip: {current_skip})...")
            
            for i, raw_job in enumerate(raw_jobs, 1):
                raw_id = raw_job.get('id')
                title = raw_job.get('raw_title', 'Unknown Title')
                company = raw_job.get('raw_company', 'Unknown Company')
                
                print(f"\n[{i}/{len(raw_jobs)}] Inspecting ID: {raw_id}")
                print(f"      Role: {title}")
                print(f"      Org : {company}")

                try:
                    # 2. Preprocess
                    input_text = self.preprocessor.format_input(
                        title=title,
                        company=company,
                        location=raw_job.get('raw_location'),
                        description=raw_job.get('raw_description')
                    )

                    # 3. Classify with LLM
                    result = self.classifier.classify(input_text)
                    
                    # Audit logging
                    self._log_audit(raw_id, result)
                    
                    processed_in_batch += 1

                    if result['is_valid']:
                        # 4. Prepare and Save Valid Job
                        # Extract extra metadata from payload if available
                        payload = raw_job.get('raw_payload') or {}
                        if isinstance(payload, str):
                            import json
                            try:
                                payload = json.loads(payload)
                            except:
                                payload = {}

                        job_data = {
                            "title": title[:200], # Safety truncate
                            "description": raw_job.get('raw_description'),
                            "company_name": company[:200],
                            "employment_type": payload.get('employment_type', 'full_time'),
                            "work_mode": payload.get('work_mode', 'hybrid'),
                            "source": "email_bot_llm_local",
                            "external_id": str(payload.get('post_id') or raw_job.get('source_uid') or raw_id),
                            "location": raw_job.get('raw_location'),
                            "country": payload.get('country', 'USA'),
                            "url": payload.get('url') or payload.get('link') or "",
                            "raw_position_id": raw_id,
                            "confidence_score": result['score'],
                            "classification_label": result['label']
                        }
                        
                        if not self.dry_run:
                            save_success = self.persistence.save_valid_job(job_data)
                            if save_success:
                                logger.info(f"       Saved to job_listing table with full metadata")
                                # 5. Mark as processed ONLY after successful save
                                status_success = self.persistence.update_raw_status(raw_id, "parsed")
                                if status_success:
                                    logger.info(f"       Status marked as 'parsed'")
                            else:
                                logger.error(f"       Failed to persist job. Status remains 'new'.")
                        else:
                            print(f"      [DRY RUN] Would save to database with following metadata:")
                            print(f"                Ext ID: {job_data['external_id']}")
                            print(f"                URL   : {job_data['url']}")
                            print(f"                Mode  : {job_data['work_mode']}")
                    else:
                        # Even if junk, we mark as parsed so we don't pick it up again
                        if not self.dry_run:
                            status_success = self.persistence.update_raw_status(raw_id, "parsed")
                            if status_success:
                                logger.info(f" Junk filtered and marked as 'parsed'")
                        else:
                            print(f" [DRY RUN] Would mark as 'parsed' (junk).")
                        
                except Exception as e:
                    logger.error(f" Error processing ID {raw_id}: {e}")
                    continue
            
            # Smart Pagination: Always move forward by the number of records we looked at
            # This ensures we don't get stuck on the same page of already-parsed records
            # if the API's processing_status filter is failing.
            current_skip += len(raw_jobs)
            
            if self.dry_run:
                print("\n" + "="*60)
                print(" DRY RUN COMPLETE - Review logs for accuracy.")
                print("="*60 + "\n")
                break
            
            time.sleep(1)

    def _log_audit(self, raw_id: int, result: dict):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        reasoning = result.get('reasoning', 'N/A').replace('\n', ' ')
        entry = (
            f"{timestamp} | ID: {raw_id:6} | Label: {result['label']:10} | "
            f"Score: {result['score']:.2f} | Reasoning: {reasoning[:100]}...\n"
        )
        try:
            with open(self.audit_log, "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

def main():
    parser = argparse.ArgumentParser(description="Classify raw job listings using Local LLM")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing to DB/API")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of records per batch (LLM is slower than BERT)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()
    
    orchestrator = LLMJobClassifyOrchestrator(
        dry_run=args.dry_run, 
        batch_size=args.batch_size,
        threshold=args.threshold
    )
    orchestrator.run()

if __name__ == "__main__":
    main()
