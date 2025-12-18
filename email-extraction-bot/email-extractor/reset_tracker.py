#!/usr/bin/env python3
"""
Reset UID Tracker - Force re-processing of emails

This script gives you options to:
1. Reset specific account (re-process all emails for that account)
2. Reset all accounts (re-process everything)
3. View current status
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def load_tracker(tracker_file):
    """Load tracker data"""
    path = Path(tracker_file)
    if not path.exists():
        return None, {}
    
    try:
        with open(path, 'r') as f:
            return path, json.load(f)
    except:
        return path, {}

def save_tracker(path, data):
    """Save tracker data"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)

def main():
    print("=" * 80)
    print("UID TRACKER RESET TOOL")
    print("=" * 80)
    
    # Check for tracker files
    trackers = {}
    for filename in ['last_run.json', 'last_run_test.json']:
        path, data = load_tracker(filename)
        if path:
            trackers[filename] = (path, data)
    
    if not trackers:
        print("\n✅ No tracker files found - system will process all emails on next run")
        return
    
    print(f"\nFound {len(trackers)} tracker file(s):\n")
    
    for filename, (path, data) in trackers.items():
        print(f"📄 {filename}: {len(data)} account(s)")
        for email, info in data.items():
            last_run = info.get('last_run', 'unknown')
            try:
                run_time = datetime.fromisoformat(last_run)
                days_ago = (datetime.now() - run_time).days
                print(f"   - {email} (last run: {days_ago} days ago)")
            except:
                print(f"   - {email} (last run: {last_run})")
    
    print("\n" + "=" * 80)
    print("RESET OPTIONS")
    print("=" * 80)
    print("\n1. Reset SPECIFIC account (re-process all emails for one account)")
    print("2. Reset ALL accounts (re-process everything)")
    print("3. Delete tracker file entirely")
    print("4. Exit (no changes)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # Reset specific account
        print("\nAvailable accounts:")
        all_emails = []
        for filename, (path, data) in trackers.items():
            for email in data.keys():
                all_emails.append((email, filename, path, data))
                print(f"  {len(all_emails)}. {email} (in {filename})")
        
        if not all_emails:
            print("No accounts found!")
            return
        
        account_choice = input(f"\nEnter account number (1-{len(all_emails)}): ").strip()
        try:
            idx = int(account_choice) - 1
            if 0 <= idx < len(all_emails):
                email, filename, path, data = all_emails[idx]
                del data[email]
                save_tracker(path, data)
                print(f"\n✅ Reset {email} in {filename}")
                print(f"   Next run will process ALL emails for this account")
            else:
                print("Invalid choice")
        except:
            print("Invalid input")
    
    elif choice == '2':
        # Reset all
        confirm = input("\n⚠️  This will reset ALL accounts. Continue? (yes/no): ").strip().lower()
        if confirm == 'yes':
            for filename, (path, data) in trackers.items():
                save_tracker(path, {})
                print(f"✅ Reset all accounts in {filename}")
            print("\n✅ All accounts reset - next run will process ALL emails")
        else:
            print("Cancelled")
    
    elif choice == '3':
        # Delete tracker file
        print("\nAvailable tracker files:")
        for i, filename in enumerate(trackers.keys(), 1):
            print(f"  {i}. {filename}")
        
        file_choice = input(f"\nEnter file number to DELETE (1-{len(trackers)}): ").strip()
        try:
            idx = int(file_choice) - 1
            filename = list(trackers.keys())[idx]
            path = trackers[filename][0]
            
            confirm = input(f"\n⚠️  Delete {filename}? (yes/no): ").strip().lower()
            if confirm == 'yes':
                path.unlink()
                print(f"✅ Deleted {filename}")
                print(f"   Next run will process ALL emails for all accounts")
            else:
                print("Cancelled")
        except:
            print("Invalid input")
    
    elif choice == '4':
        print("\nNo changes made")
    
    else:
        print("\nInvalid choice")

if __name__ == "__main__":
    main()
