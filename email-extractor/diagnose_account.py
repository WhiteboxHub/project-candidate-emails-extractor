#!/usr/bin/env python3
"""
Diagnostic script to check why email fetching stopped
"""

import json
import os
from pathlib import Path
from datetime import datetime

def check_tracker_status():
    """Check the UID tracker status"""
    print("=" * 80)
    print("EMAIL EXTRACTOR DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check for tracker files
    tracker_files = ['last_run.json', 'last_run_test.json']
    
    for tracker_file in tracker_files:
        tracker_path = Path(tracker_file)
        
        print(f"\n{'─' * 80}")
        print(f"Checking: {tracker_file}")
        print(f"{'─' * 80}")
        
        if not tracker_path.exists():
            print(f"[NOT FOUND] File does not exist")
            print(f"   → This means: First run OR file was deleted")
            print(f"   → Action: Will process ALL emails on next run")
            continue
        
        try:
            with open(tracker_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                print(f"[WARNING] File is EMPTY")
                print(f"   → This means: No accounts have been processed yet")
                continue
            
            print(f"[OK] File found with {len(data)} account(s)\n")
            
            for email, info in data.items():
                last_uid = info.get('last_uid', 'unknown')
                last_run = info.get('last_run', 'unknown')
                
                try:
                    run_time = datetime.fromisoformat(last_run)
                    time_ago = datetime.now() - run_time
                    days_ago = time_ago.days
                    hours_ago = time_ago.seconds // 3600
                    
                    print(f"Account: {email}")
                    print(f"  Last UID processed: {last_uid}")
                    print(f"  Last run: {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Time since last run: {days_ago} days, {hours_ago} hours ago")
                    
                    if days_ago > 7:
                        print(f"  [WARNING] Last run was over a week ago!")
                    elif days_ago > 1:
                        print(f"  [INFO] Last run was {days_ago} days ago")
                    else:
                        print(f"  [OK] Recent run")
                    
                    print()
                    
                except Exception as e:
                    print(f"Account: {email}")
                    print(f"  Last UID: {last_uid}")
                    print(f"  Last run: {last_run}")
                    print(f"  [WARNING] Could not parse date: {e}\n")
        
        except json.JSONDecodeError as e:
            print(f"[ERROR] INVALID JSON in {tracker_file}")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"[ERROR] reading {tracker_file}")
            print(f"   Error: {e}")
    
    print(f"\n{'=' * 80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'=' * 80}\n")
    
    print("Why might the extractor not be fetching new emails?\n")
    print("1. [OK] WORKING AS DESIGNED: No new emails since last run")
    print("   - The UID tracker stores the last processed email UID")
    print("   - On next run, it only fetches emails with UID > last_uid")
    print("   - If mailbox has no new emails, nothing to fetch!\n")
    
    print("2. [WARNING] STUCK ON OLD UID: Tracker has stale data")
    print("   - If last run was long ago but you KNOW new emails exist")
    print("   - Solution: Delete the tracker file to force full re-scan\n")
    
    print("3. [ERROR] CONNECTION ISSUE: Can't connect to mailbox")
    print("   - Wrong credentials, network issues, IMAP disabled")
    print("   - Check service.py logs for connection errors\n")
    
    print("4. [ERROR] DATABASE ISSUE: Can't fetch account credentials")
    print("   - Database connection problems")
    print("   - Account missing from candidate table\n")
    
    print(f"{'=' * 80}")
    print("RECOMMENDED ACTIONS")
    print(f"{'=' * 80}\n")
    
    print("To force re-processing of ALL emails:")
    print("  1. Delete last_run.json (or last_run_test.json)")
    print("  2. Run the service again\n")
    
    print("To check if new emails exist:")
    print("  1. Manually log into the email account")
    print("  2. Check if emails arrived after the 'Last run' timestamp above\n")
    
    print("To see detailed logs:")
    print("  1. Run: python3 service.py")
    print("  2. Look for 'No new emails to process' message")
    print("  3. Check for connection or authentication errors\n")

if __name__ == "__main__":
    check_tracker_status()
