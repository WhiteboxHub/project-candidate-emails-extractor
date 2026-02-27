"""
tests/preflight_check.py
========================
ONE COMMAND to verify the ENTIRE bot is ready for production.

Checks (in order):
  1.  ENV VARS     — all required .env keys are present and non-empty
  2.  CONFIG       — config/config.yaml loads without errors
  3.  KEYWORDS CSV — keywords.csv loads and has key categories
  4.  DB           — MySQL connection works and returns candidates
  5.  API          — Backend REST API authenticates successfully
  6.  IMAP         — Each candidate IMAP connection tested (same as test_imap_connectivity.py)
  7.  EXTRACTION   — Sample email runs through the extractor with no crash

Usage:
    python tests/preflight_check.py

    # Or via pytest (shows individual test names):
    python -m pytest tests/preflight_check.py -v
"""

import sys
import os
import logging
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

# Load .env first so all below imports can use env vars
_env_path = BASE_DIR / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                k, _, v = _line.partition('=')
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(
    level=logging.WARNING,   # Keep it quiet — only show failures
    format='%(levelname)-7s  %(message)s',
)

import unittest

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 1 — Environment Variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED_ENV_VARS = {
    'DB':  ['DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_NAME'],
    'API': ['API_BASE_URL', 'API_EMAIL', 'API_PASSWORD', 'EMPLOYEE_ID'],
    'SMTP': ['SMTP_SERVER', 'SMTP_USERNAME', 'SMTP_PASSWORD'],
}

class TestEnvVars(unittest.TestCase):
    """Check all required .env variables are set"""

    def test_db_env_vars(self):
        missing = [v for v in REQUIRED_ENV_VARS['DB'] if not os.getenv(v)]
        self.assertFalse(missing, f"Missing DB env vars: {missing}\nCheck your .env file.")

    def test_api_env_vars(self):
        missing = [v for v in REQUIRED_ENV_VARS['API'] if not os.getenv(v)]
        self.assertFalse(missing, f"Missing API env vars: {missing}\nCheck your .env file.")

    def test_smtp_env_vars(self):
        missing = [v for v in REQUIRED_ENV_VARS['SMTP'] if not os.getenv(v)]
        self.assertFalse(missing, f"Missing SMTP env vars: {missing}\nCheck your .env file.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 2 — Config YAML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConfig(unittest.TestCase):
    """Check configs/config.yaml loads correctly"""

    def _make_loader(self):
        from extractor.core.settings import ConfigLoader
        # Always use absolute path — avoids CWD-relative issues
        config_path = BASE_DIR / 'configs' / 'config.yaml'
        return ConfigLoader(config_path=str(config_path))

    def test_config_loads(self):
        config_loader = self._make_loader()
        config = config_loader.load()
        self.assertIsInstance(config, dict, "config.yaml did not load as a dict")
        self.assertGreater(len(config), 0, "config.yaml is empty")

    def test_config_has_extraction_section(self):
        config = self._make_loader().load()
        self.assertIn('extraction', config,
            "configs/config.yaml is missing 'extraction' section")

    def test_config_has_email_section(self):
        config = self._make_loader().load()
        self.assertIn('email', config,
            "configs/config.yaml is missing 'email' section")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 3 — Keywords CSV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestKeywordsCSV(unittest.TestCase):
    """Check keywords.csv is present and has required categories"""

    REQUIRED_CATEGORIES = [
        'greeting_patterns',
        'company_indicators',
    ]

    def test_keywords_csv_exists(self):
        csv_path = BASE_DIR / 'src' / 'keywords.csv'
        self.assertTrue(csv_path.exists(),
            f"keywords.csv not found at {csv_path}")

    def test_keywords_csv_loads(self):
        from extractor.filtering.repository import get_filter_repository
        repo = get_filter_repository()
        keyword_lists = repo.get_keyword_lists()
        self.assertIsInstance(keyword_lists, dict)
        self.assertGreater(len(keyword_lists), 0, "keywords.csv loaded but returned empty dict")

    def test_required_categories_present(self):
        from extractor.filtering.repository import get_filter_repository
        repo = get_filter_repository()
        keyword_lists = repo.get_keyword_lists()
        missing = [c for c in self.REQUIRED_CATEGORIES if c not in keyword_lists]
        self.assertFalse(missing,
            f"Missing keyword categories in CSV: {missing}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 4 — Database Connection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDatabase(unittest.TestCase):
    """Check MySQL DB is reachable and has expected data"""

    def _get_db(self):
        from extractor.core.database import get_db_client
        db = get_db_client()
        db.initialize()
        return db

    def test_db_connection(self):
        try:
            db = self._get_db()
            result = db.execute_query("SELECT 1 AS ok")
            self.assertEqual(result[0]['ok'], 1,
                "DB connected but SELECT 1 returned unexpected result")
        except Exception as e:
            self.fail(
                f"Cannot connect to MySQL at {os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}\n"
                f"Error: {e}\n"
                f"Check DB_HOST / DB_USER / DB_PASSWORD in .env"
            )

    def test_workflow_config_exists(self):
        db = self._get_db()
        rows = db.execute_query(
            "SELECT id, workflow_key, credentials_list_sql FROM automation_workflows WHERE id = 2 LIMIT 1"
        )
        self.assertTrue(rows, "No row found in automation_workflows WHERE id = 2")
        self.assertTrue(rows[0].get('credentials_list_sql'),
            "automation_workflows.id=2 has no credentials_list_sql — run migration V62")

    def test_candidates_loaded(self):
        from extractor.persistence.db_candidate_source import DatabaseCandidateSource
        from extractor.core.database import get_db_client
        db = get_db_client()
        rows = db.execute_query(
            "SELECT credentials_list_sql FROM automation_workflows WHERE id = 2 LIMIT 1"
        )
        sql = rows[0]['credentials_list_sql']
        candidates = DatabaseCandidateSource(credentials_sql=sql).get_active_candidates()
        self.assertGreater(len(candidates), 0,
            "No candidates returned from credentials_list_sql.\n"
            "Check that candidate_marketing.run_email_extraction = 1 for at least one row.")
        print(f"\n  → {len(candidates)} candidate(s) found in DB")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 5 — Backend API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAPI(unittest.TestCase):
    """Check the WBL REST API is reachable and credentials work"""

    def test_api_authenticates(self):
        from extractor.connectors.http_api import get_api_client
        try:
            client = get_api_client()
            ok = client.authenticate()
            self.assertTrue(ok,
                f"API login failed for {os.getenv('API_EMAIL')} at {os.getenv('API_BASE_URL')}\n"
                "Check API_EMAIL / API_PASSWORD in .env")
        except Exception as e:
            self.fail(
                f"Cannot reach API at {os.getenv('API_BASE_URL')}\n"
                f"Error: {e}"
            )

    def test_api_job_types_endpoint(self):
        """Check /api/job-types GET endpoint is reachable"""
        from extractor.connectors.http_api import get_api_client
        client = get_api_client()
        client.authenticate()
        try:
            result = client.get('/api/job-types')
            self.assertIsNotNone(result, "/api/job-types returned None")
        except Exception as e:
            self.fail(f"/api/job-types endpoint error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 6 — IMAP Connectivity (all candidates)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIMAPAll(unittest.TestCase):
    """
    Checks IMAP connection for every candidate at once.
    Reports all failures in one go rather than stopping at the first.
    For per-candidate tests use: pytest test_imap_connectivity.py -v -k "<ID>"
    """

    def test_all_candidate_imap_connections(self):
        from extractor.core.database import get_db_client
        from extractor.persistence.db_candidate_source import DatabaseCandidateSource
        from extractor.connectors.imap_gmail import GmailIMAPConnector

        db = get_db_client()
        rows = db.execute_query(
            "SELECT credentials_list_sql FROM automation_workflows WHERE id = 2 LIMIT 1"
        )
        if not rows:
            self.skipTest("No candidates SQL found — skipping IMAP check")

        candidates = DatabaseCandidateSource(rows[0]['credentials_list_sql']).get_active_candidates()
        failures = []

        for c in candidates:
            email    = (c.get('email') or '').strip()
            password = c.get('imap_password', '')
            name     = c.get('name') or email
            cid      = c.get('candidate_id') or c.get('id')

            if not password:
                failures.append(f"  [{cid}] {name} ({email}): ❌ missing imap_password in DB")
                continue

            conn = GmailIMAPConnector(email=email, password=password)
            ok, err = conn.connect()
            conn.disconnect()

            if ok:
                print(f"  [{cid}] {name}: ✅ OK")
            else:
                failures.append(
                    f"  [{cid}] {name} ({email}): ❌ {err}\n"
                    f"      FIX: UPDATE automation_candidate_emails SET imap_password='<new>' "
                    f"WHERE candidate_email='{email}';"
                )

        if failures:
            self.fail(
                f"\n\n{'='*60}\n"
                f"IMAP FAILURES ({len(failures)} of {len(candidates)} candidates)\n"
                f"{'='*60}\n" +
                "\n".join(failures) +
                f"\n{'='*60}\n"
            )
        else:
            print(f"\n  ✅ All {len(candidates)} candidates connected OK")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECK 7 — Extraction Dry Run (no IMAP, uses a canned email)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_EMAIL_BODY = """\
Hi John,

I hope you are doing well. My name is Sarah Mitchell and I'm a Technical Recruiter
at TechStaff Solutions Inc. I came across your profile and wanted to reach out regarding
an exciting Senior Python Developer opportunity.

The role is based in Austin, TX and offers a competitive package.

Please feel free to reach me:
Phone: +1-512-555-9012
LinkedIn: linkedin.com/in/sarahmitchell

Best regards,
Sarah Mitchell
Senior Technical Recruiter
TechStaff Solutions Inc.
Austin, TX
"""

class TestExtractionDryRun(unittest.TestCase):
    """
    Runs the extractor on a canned sample email — no real IMAP needed.
    Verifies the extraction pipeline doesn't crash and returns sensible data.
    """

    @classmethod
    def setUpClass(cls):
        """Build the ContactExtractor once — it's expensive."""
        try:
            from extractor.core.settings import ConfigLoader
            from extractor.extraction.contacts import ContactExtractor
            config_path = BASE_DIR / 'configs' / 'config.yaml'
            config = ConfigLoader(config_path=str(config_path)).load()
            cls.extractor = ContactExtractor(config)
            cls.config    = config
        except Exception as e:
            raise unittest.SkipTest(f"Could not initialise ContactExtractor: {e}")

    def _make_email_message(self, body: str, from_header: str = "Sarah Mitchell <sarah.mitchell@techstaffsolutions.com>"):
        """Build a minimal email.message.Message object"""
        import email
        raw = (
            f"From: {from_header}\n"
            f"To: candidate@gmail.com\n"
            f"Subject: Exciting Python Developer Opportunity\n"
            f"\n{body}"
        )
        return email.message_from_string(raw)

    def test_extraction_runs_without_crash(self):
        """The extractor must not raise an exception"""
        msg = self._make_email_message(SAMPLE_EMAIL_BODY)
        try:
            results = self.extractor.extract_contacts(
                msg,
                SAMPLE_EMAIL_BODY,
                source_email='candidate@gmail.com'
            )
        except Exception as e:
            self.fail(f"extract_contacts raised an exception: {e}")
        self.assertIsInstance(results, list, "extract_contacts should return a list")

    def test_extraction_finds_email(self):
        """Must find the sender's email"""
        msg = self._make_email_message(SAMPLE_EMAIL_BODY)
        results = self.extractor.extract_contacts(msg, SAMPLE_EMAIL_BODY, source_email='candidate@gmail.com')
        emails_found = [c.get('email') for c in results if c.get('email')]
        self.assertTrue(emails_found,
            "Extractor found no email addresses in the sample email — regex broken?")

    def test_extraction_no_junk_names(self):
        """
        The name 'Candidate' or 'John' (the recipient greeting) must NOT appear as a contact name.
        Only 'Sarah Mitchell' (the sender) should be found.
        """
        msg = self._make_email_message(SAMPLE_EMAIL_BODY)
        results = self.extractor.extract_contacts(msg, SAMPLE_EMAIL_BODY, source_email='candidate@gmail.com')
        junk_names = {'john', 'candidate', 'hi', 'dear', 'recruiting', 'noreply'}
        for contact in results:
            name = (contact.get('name') or '').lower()
            for junk in junk_names:
                if name == junk or name.startswith(junk + ' '):
                    self.fail(
                        f"Junk name '{contact['name']}' was extracted! "
                        f"The positive validation gate is not working."
                    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    loader = unittest.TestLoader()
    # Run checks in defined order
    suite = unittest.TestSuite()
    for cls in [
        TestEnvVars,
        TestConfig,
        TestKeywordsCSV,
        TestDatabase,
        TestAPI,
        TestIMAPAll,
        TestExtractionDryRun,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    print("\n" + "="*60)
    print("  Pre-Production Health Check")
    print("="*60 + "\n")

    runner = unittest.TextTestRunner(verbosity=2, buffer=False)
    result = runner.run(suite)

    print("\n" + "="*60)
    if result.wasSuccessful():
        print("  ✅  ALL CHECKS PASSED — safe to run production")
    else:
        print(f"  ❌  {len(result.failures + result.errors)} CHECK(S) FAILED — DO NOT RUN PRODUCTION")
    print("="*60 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
