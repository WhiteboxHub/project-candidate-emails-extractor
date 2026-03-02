"""Check CSV state and deduplicate by ID if needed."""
import csv
from pathlib import Path

csv_path = Path('src/keywords.csv').resolve()
rows = list(csv.DictReader(open(csv_path, encoding='utf-8')))
print(f'Total rows before dedup: {len(rows)}')

# Deduplicate by ID — keep last occurrence
seen = {}
for r in rows:
    seen[r['id']] = r
deduped = list(seen.values())
print(f'Total rows after dedup:  {len(deduped)}')

new_cats = ['junk_name_patterns', 'recruiter_email_signals', 'non_recruiter_body_signals']
for c in new_cats:
    found = any(r['category'] == c for r in deduped)
    print(f'  {c}: {"OK" if found else "MISSING"}')

# Rewrite the file cleanly if duplicates were found
if len(deduped) < len(rows):
    print('Writing deduplicated CSV...')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)
    print('Done.')
else:
    print('No duplicates found - CSV is clean.')
