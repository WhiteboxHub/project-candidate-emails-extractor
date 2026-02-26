"""
test_imap_connectivity.py
=========================
Pre-production IMAP connectivity health-check for all candidates.

Loads credentials the SAME way production does:
  automation_workflows.credentials_list_sql → DB → GmailIMAPConnector

Env vars (DB_HOST, DB_PORT, etc.) are loaded from .env by conftest.py
BEFORE pytest collects any tests.

Run commands:
    # All candidates:
    python -m pytest src/extractor/tests/test_imap_connectivity.py -v

    # One candidate by ID:
    python -m pytest src/extractor/tests/test_imap_connectivity.py -v -k "752"

    # One candidate by email prefix:
    python -m pytest src/extractor/tests/test_imap_connectivity.py -v -k "sheela"

    # Standalone (no pytest):
    python src/extractor/tests/test_imap_connectivity.py
"""
import sys
import os
import logging
import unittest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
)
logger = logging.getLogger(__name__)

# ── Make src/ importable when run standalone ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ════════════════════════════════════════════════════════════════════════════
#  PYTEST test  (parametrized by conftest.py → pytest_generate_tests)
# ════════════════════════════════════════════════════════════════════════════

def test_imap_candidate(imap_candidate):
    """
    IMAP connectivity check for a single candidate.
    The `imap_candidate` fixture is injected and parametrized
    by conftest.pytest_generate_tests — one test per DB row.
    """
    if imap_candidate is None:
        raise RuntimeError(
            "No candidates loaded from DB.\n"
            "Check: DB_HOST / DB_USER / DB_PASSWORD in your .env file\n"
            "and that automation_workflows.id=2 has a credentials_list_sql value."
        )

    from extractor.connectors.imap_gmail import GmailIMAPConnector

    email    = (imap_candidate.get('email') or '').strip()
    password = imap_candidate.get('imap_password', '')
    name     = imap_candidate.get('name') or email
    cid      = imap_candidate.get('candidate_id') or imap_candidate.get('id')

    assert email,    f"[{cid}] {name}: missing email in DB"
    assert password, (
        f"[{cid}] {name} ({email}): missing imap_password in DB\n"
        f"Fix: UPDATE automation_candidate_emails "
        f"SET imap_password='<app-password>' WHERE candidate_email='{email}';"
    )

    connector = GmailIMAPConnector(email=email, password=password)
    ok, err = connector.connect()

    if ok:
        connector.disconnect()
        logger.info(f"[{cid}] {name} ({email}): ✅ Connected OK")
    else:
        connector.disconnect()
        raise AssertionError(
            f"\n\n{'='*60}\n"
            f"IMAP AUTHENTICATION FAILED\n"
            f"{'='*60}\n"
            f"  Candidate : [{cid}] {name}\n"
            f"  Email     : {email}\n"
            f"  Error     : {err}\n\n"
            f"FIX — ask the candidate to:\n"
            f"  1. Go to myaccount.google.com → Security → 2-Step Verification\n"
            f"  2. Generate a new App Password (Mail · Other device)\n"
            f"  3. Run this SQL:\n"
            f"     UPDATE automation_candidate_emails\n"
            f"     SET imap_password = '<new-password>'\n"
            f"     WHERE candidate_email = '{email}';\n"
            f"{'='*60}\n"
        )


# ════════════════════════════════════════════════════════════════════════════
#  STANDALONE runner  (python test_imap_connectivity.py)
# ════════════════════════════════════════════════════════════════════════════

def _standalone_load_candidates():
    # Load .env manually when running standalone
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, _, v = line.partition('=')
                    os.environ.setdefault(k.strip(), v.strip())

    try:
        from extractor.persistence.db_candidate_source import DatabaseCandidateSource
        from extractor.core.database import get_db_client

        db = get_db_client()
        rows = db.execute_query(
            "SELECT credentials_list_sql FROM automation_workflows WHERE id = 2 LIMIT 1"
        )
        if not rows or not rows[0].get('credentials_list_sql'):
            print("❌  No credentials_list_sql in automation_workflows id=2")
            return []
        sql = rows[0]['credentials_list_sql']
        return DatabaseCandidateSource(credentials_sql=sql).get_active_candidates()
    except Exception as exc:
        print(f"❌  Could not load candidates: {exc}")
        return []


if __name__ == '__main__':
    from extractor.connectors.imap_gmail import GmailIMAPConnector

    print("\n" + "="*60)
    print("  IMAP Pre-flight Connectivity Check")
    print("="*60)

    candidates = _standalone_load_candidates()
    if not candidates:
        print("No candidates loaded.")
        sys.exit(1)

    print(f"\nChecking {len(candidates)} candidate(s)...\n")
    passed, failed = [], []

    for c in candidates:
        email    = (c.get('email') or '').strip()
        password = c.get('imap_password', '')
        name     = c.get('name') or email
        cid      = c.get('candidate_id') or c.get('id')

        if not password:
            msg = f"  ❌  [{cid}] {name} ({email}): missing imap_password"
            print(msg); failed.append(msg); continue

        conn = GmailIMAPConnector(email=email, password=password)
        ok, err = conn.connect()
        conn.disconnect()

        if ok:
            msg = f"  ✅  [{cid}] {name} ({email})"
            print(msg); passed.append(msg)
        else:
            msg = f"  ❌  [{cid}] {name} ({email}): {err}"
            print(msg); failed.append(msg)

    print("\n" + "="*60)
    print(f"  Results: {len(passed)} OK  |  {len(failed)} FAILED")
    print("="*60)
    if failed:
        print("\nFailed:")
        for f in failed:
            print(f)
        sys.exit(1)
    else:
        print("\n✅  All candidates connected successfully.\n")
        sys.exit(0)
