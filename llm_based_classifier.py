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
    def __init__(self, dry_run: bool = False, batch_size: int = 15, threshold: float = 0.7):
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
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'PRODUCTION'} | Batch: {self.batch_size}")
        
        while True:
            # 1. Fetch raw jobs
            raw_jobs = self.persistence.fetch_raw_jobs(limit=self.batch_size)
            if not raw_jobs:
                print("\n" + "-"*60)
                print(" No new jobs to process. System idling.")
                print("-"*60 + "\n")
                break
            
            print(f"\nProcessing batch of {len(raw_jobs)} candidates...")
            
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
                    
                    if result['is_valid']:
                        # 4. Prepare and Save Valid Job
                        job_data = {
                            "title": title[:200], # Safety truncate
                            "company_name": company[:200],
                            "location": raw_job.get('raw_location'),
                            "source": "email_bot_llm_local",
                            "raw_position_id": raw_id,
                            "confidence_score": result['score'],
                            "classification_label": result['label']
                        }
                        
                        if not self.dry_run:
                            save_success = self.persistence.save_valid_job(job_data)
                            if save_success:
                                logger.info(f"       Saved to job_listing table")
                                # 5. Mark as processed ONLY after successful save
                                status_success = self.persistence.update_raw_status(raw_id, "parsed")
                                if status_success:
                                    logger.info(f"       Status marked as 'parsed'")
                            else:
                                logger.error(f"       Failed to persist job. Status remains 'new'.")
                        else:
                            print(f"      [DRY RUN] Would save to database.")
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
