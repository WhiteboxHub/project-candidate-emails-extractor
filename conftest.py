"""
conftest.py  (project root)
============================
Loads the .env file BEFORE pytest collects any tests.
This ensures DB_HOST, DB_USER, DB_PASSWORD etc. are available
when test_imap_connectivity.py calls DatabaseCandidateSource.

Also provides the `imap_candidate` fixture used by the IMAP test.
"""
import os
import sys
import logging

# ── Load .env from the project root before collection ───────────────────────
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _, _val = _line.partition('=')
                os.environ.setdefault(_key.strip(), _val.strip())
    logging.getLogger(__name__).debug(f"conftest: loaded {_env_path}")

# ── Ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def _load_candidates():
    """Load all active candidates from DB using the workflow's SQL."""
    try:
        from extractor.persistence.db_candidate_source import DatabaseCandidateSource
        from extractor.core.database import get_db_client

        db = get_db_client()
        rows = db.execute_query(
            "SELECT credentials_list_sql FROM automation_workflows WHERE id = 2 LIMIT 1"
        )
        if not rows or not rows[0].get('credentials_list_sql'):
            logging.warning("conftest: No credentials_list_sql in automation_workflows id=2")
            return []

        sql = rows[0]['credentials_list_sql']
        source = DatabaseCandidateSource(credentials_sql=sql)
        return source.get_active_candidates()

    except Exception as exc:
        logging.warning(f"conftest: Could not load candidates from DB — {exc}")
        return []


def pytest_generate_tests(metafunc):
    """
    Parametrize any test that requests the `imap_candidate` fixture.
    Each candidate becomes a separate test case named by candidate_id + email.
    """
    if 'imap_candidate' in metafunc.fixturenames:
        candidates = _load_candidates()
        if not candidates:
            # Still show one test so the user sees the failure reason
            metafunc.parametrize('imap_candidate', [None], ids=['no_candidates_loaded'])
        else:
            ids = []
            for c in candidates:
                cid   = c.get('candidate_id') or c.get('id') or 'x'
                email = (c.get('email') or 'unknown').split('@')[0]
                ids.append(f"{cid}_{email}")
            metafunc.parametrize('imap_candidate', candidates, ids=ids)
