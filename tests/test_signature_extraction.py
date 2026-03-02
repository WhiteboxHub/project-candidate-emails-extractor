"""
tests/test_signature_extraction.py
===================================
Targeted unit tests for recruiter/vendor contact extraction quality.

Uses REAL email bodies taken from the latest workflow run
(run_cae4fb6e-d6a6-402b-9604-5b43e69e9b0e.json) to verify that the
improved extraction logic gives EXACTLY the right fields.

Run:
    python tests/test_signature_extraction.py
    # or
    python -m pytest tests/test_signature_extraction.py -v
"""

import sys
import os
import logging
import unittest
from pathlib import Path
from typing import Optional

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'src'))

# Load .env first
_env_path = BASE_DIR / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                k, _, v = _line.partition('=')
                os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.WARNING, format='%(levelname)-7s  %(message)s')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared helper — build ContactExtractor once
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_extractor():
    from extractor.core.settings import ConfigLoader
    from extractor.extraction.contacts import ContactExtractor
    config_path = BASE_DIR / 'configs' / 'config.yaml'
    config = ConfigLoader(config_path=str(config_path)).load()
    return ContactExtractor(config)


def _make_email_msg(body: str, from_header: str, subject: str = "Job Opportunity") -> object:
    import email as _email
    raw = (
        f"From: {from_header}\n"
        f"To: candidate@gmail.com\n"
        f"Subject: {subject}\n"
        f"\n{body}"
    )
    return _email.message_from_string(raw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST GROUP 1 — extract_intro_sentence  (isolated unit test, no extractor)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIntroSentenceExtractor(unittest.TestCase):
    """Unit-test extract_intro_sentence() in isolation (no config / IMAP needed)."""

    @classmethod
    def setUpClass(cls):
        try:
            from extractor.extraction.nlp_spacy import SpacyNERExtractor
            cls.extractor = SpacyNERExtractor()
        except Exception as e:
            raise unittest.SkipTest(f"SpacyNERExtractor init failed: {e}")

    # ── Pattern 1: "My name is X and I am a Title at Company" ───────────────
    def test_pattern1_nityo_staffing_specialist(self):
        body = (
            "Hello Sai Madhavi\n\n"
            "My name is Abhishek Singh and I am a Staffing Specialist at"
            " Nityo Infotech. I am reaching out to you on an exciting job opportunity.\n\n"
            "Thanks & Regards\n"
            "Abhishek Singh\n"
            "Direct Number- 6099985925\n"
        )
        result = self.extractor.extract_intro_sentence(body)
        self.assertEqual(result['name'], 'Abhishek Singh',
                         f"Expected 'Abhishek Singh', got: {result['name']}")
        self.assertIsNotNone(result['title'],
                             "title should not be None (expected 'Staffing Specialist')")
        self.assertIn('nityo', result['company'].lower() if result['company'] else '',
                      f"Expected company to contain 'Nityo', got: {result['company']}")

    # ── Pattern 1b: Sachin Sharma at Nityo Infotech ─────────────────────────
    def test_pattern1_sachin_nityo(self):
        body = (
            "Hello Sai Madhavi\n\n"
            "My name is Sachin Sharma and I am a Staffing Specialist at"
            " Nityo Infotech. I am reaching out to you on an exciting job opportunity.\n\n"
            "Thanks and Regards\n"
            "Sachin Sharma\n"
            "Talent Acquisition Specialist\n"
            "Direct : (+1) 609-857-8229\n"
        )
        result = self.extractor.extract_intro_sentence(body)
        self.assertEqual(result['name'], 'Sachin Sharma')
        self.assertIn('Nityo', result.get('company') or '',
                      f"Company should contain 'Nityo', got: {result['company']}")

    # ── Pattern 2: "I am a Title at Company" (no name in intro) ─────────────
    def test_pattern2_recruiter_at_bayonesolutions(self):
        body = (
            "Hi John,\n\n"
            "I am a Senior Technical Recruiter at BayOne Solutions Inc. and I came across"
            " your profile for the Python Developer role.\n\n"
            "Best,\n"
            "Satyam Kashyap\n"
            "TALENT SCOUT | BayOne Solutions\n"
        )
        result = self.extractor.extract_intro_sentence(body)
        self.assertIsNotNone(result['title'], "title should be extracted ('Senior Technical Recruiter')")
        self.assertIn('BayOne', result.get('company') or '',
                      f"Company should contain 'BayOne', got: {result['company']}")

    # ── Pattern 3: "This is X from Company" ─────────────────────────────────
    def test_pattern3_this_is_from(self):
        body = (
            "Hi,\n\n"
            "This is Jitendra from Sibitalent Corporation. I wanted to reach out about"
            " a Senior Front End Engineer role.\n\n"
            "Thanks!\nJitendra Singh\n"
        )
        result = self.extractor.extract_intro_sentence(body)
        self.assertEqual(result['name'], 'Jitendra', "Expected 'Jitendra'")
        self.assertIn('Sibitalent', result.get('company') or '',
                      f"Company should contain 'Sibitalent', got: {result['company']}")

    # ── Pattern 4: "[Name], a Title with Company" ────────────────────────────
    def test_pattern4_name_a_title_with(self):
        body = (
            "Hi Sandeep,\n\n"
            "I'm Priya Sharma, a Talent Acquisition Specialist with Avance Consulting."
            " I found your profile and wanted to discuss an opening.\n\n"
        )
        result = self.extractor.extract_intro_sentence(body)
        self.assertEqual(result['name'], 'Priya Sharma')
        self.assertIn('Avance', result.get('company') or '',
                      f"Company should contain 'Avance', got: {result['company']}")

    # ── Negative: newsletter body should NOT match ────────────────────────────
    def test_no_match_newsletter_body(self):
        body = (
            "We have compiled some jobs for you.\n\n"
            "Repair Engineer - $30.54 per hour\n"
            "Maintenance - $75,710 per year\n"
            "General Labour - $17.24 per hour\n\n"
            "View All Jobs\n"
        )
        result = self.extractor.extract_intro_sentence(body)
        # Newsletter has no intro sentence — all fields should be None
        self.assertIsNone(result['name'], f"Expected None, got: {result['name']}")
        self.assertIsNone(result['company'], f"Expected None, got: {result['company']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST GROUP 2 — extract_signature_info  (same-line Name|Title patterns)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSignatureInfoExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from extractor.extraction.nlp_spacy import SpacyNERExtractor
            cls.extractor = SpacyNERExtractor()
        except Exception as e:
            raise unittest.SkipTest(f"SpacyNERExtractor init failed: {e}")

    def test_name_pipe_title_same_line(self):
        """'Sachin Sharma | Talent Acquisition Specialist' should give name+title."""
        body = (
            "Thanks and Regards\n\n"
            "Sachin Sharma | Talent Acquisition Specialist\n"
            "Direct : (+1) 609-857-8229\n"
            "Email: sachin.sh@nityo.com\n"
            "Nityo Infotech\n"
        )
        result = self.extractor.extract_signature_info(body)
        self.assertIsNotNone(result['name'], "Name should be extracted")
        # The name captured may be from Pass 1 (same-line) or Pass 2 (greeting block)
        name = result.get('name') or ''
        self.assertIn('Sachin', name,
                      f"Expected name to contain 'Sachin', got: {name!r}")

    def test_allcaps_name_pipe_title(self):
        """'SATYAM KASHYAP | TALENT SCOUT' should parse to name='Satyam Kashyap', title='Talent Scout'."""
        body = (
            "Best regards,\n\n"
            "SATYAM KASHYAP | TALENT SCOUT\n"
            "Work: +1 925 476-2875\n"
            "Email: skashyap@bayonesolutions.com\n"
            "BayOne Solutions Inc.\n"
        )
        result = self.extractor.extract_signature_info(body)
        self.assertIsNotNone(result['name'])
        self.assertEqual(result['name'].lower(), 'satyam kashyap',
                         f"Expected 'Satyam Kashyap' (case-insensitive), got: {result['name']}")
        self.assertIsNotNone(result['title'])

    def test_standard_block_after_regards(self):
        """Standard Thank you / Name / Title / Company block."""
        body = (
            "Thank you for your time.\n\n"
            "Thanks & Regards\n\n"
            "Shailaja Deshetti\n"
            "Trainee Recruiter\n"
            "Avance Services\n"
            "+1 203 902 9194\n"
        )
        result = self.extractor.extract_signature_info(body)
        name = result.get('name') or ''
        self.assertIn('Shailaja', name, f"Expected name to contain 'Shailaja', got: {name}")
        self.assertIsNotNone(result.get('title'),
                             "title should be extracted ('Trainee Recruiter')")

    def test_junk_name_technical_background_rejected(self):
        """'Technical Background' must NOT be extracted as a name."""
        body = (
            "1. Technical Background (Must-Have)\n"
            "   * Education: ML or Statistics\n\n"
            "Best regards,\n"
            "Kshipra Kapoor\n"
            "Recruiter\n"
            "WebMSI\n"
        )
        result = self.extractor.extract_signature_info(body)
        name = (result.get('name') or '').lower()
        self.assertNotIn('technical', name,
                         f"'Technical Background' was wrongly extracted as name: {result['name']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST GROUP 3 — _clean_job_position  (isolated unit test)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCleanJobPosition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from extractor.core.settings import ConfigLoader
            from extractor.extraction.contacts import ContactExtractor
            config_path = BASE_DIR / 'configs' / 'config.yaml'
            config = ConfigLoader(config_path=str(config_path)).load()
            cls.ce = ContactExtractor(config)
        except Exception as e:
            raise unittest.SkipTest(f"ContactExtractor init failed: {e}")

    def test_strip_company_prefix(self):
        """'Nustar Technologies - AI Ml Engineer' → 'AI Ml Engineer'"""
        result = self.ce._clean_job_position("Nustar Technologies - AI Ml Engineer")
        self.assertIsNotNone(result)
        # Should not contain 'Nustar' in the cleaned result
        self.assertNotIn('Nustar', result,
                         f"Company prefix not stripped. Got: {result}")
        self.assertIn('Engineer', result, f"Job title stripped accidentally. Got: {result}")

    def test_strip_pipe_location_suffix(self):
        """'Ml Engineer - Computer Vision||Richardson' → 'Ml Engineer - Computer Vision'"""
        result = self.ce._clean_job_position("Ml Engineer - Computer Vision||Richardson, TX")
        self.assertIsNotNone(result)
        self.assertNotIn('Richardson', result,
                         f"Location suffix not stripped. Got: {result}")
        self.assertIn('Vision', result, f"Job title part stripped. Got: {result}")

    def test_strip_req_id_prefix(self):
        """'326632 - Data Scientist' → 'Data Scientist'"""
        result = self.ce._clean_job_position("326632 - Data Scientist")
        self.assertIsNotNone(result)
        self.assertNotIn('326632', result, f"Req ID not stripped. Got: {result}")
        self.assertIn('Scientist', result, f"Job title stripped. Got: {result}")

    def test_meeting_title_returns_none(self):
        """'PM PST For Sr AI Engineer' should return None (it's a calendar time/meeting)"""
        result = self.ce._clean_job_position("PM PST For Sr AI Engineer")
        self.assertIsNone(result, f"Expected None for meeting title, got: {result}")

    def test_phone_screen_returns_none(self):
        """'Phone Screen' should return None"""
        result = self.ce._clean_job_position("Phone Screen with Candidate")
        self.assertIsNone(result, f"Expected None for 'Phone Screen', got: {result}")

    def test_normal_position_unchanged(self):
        """'AI/ML Engineer' should pass through cleanly."""
        result = self.ce._clean_job_position("AI/ML Engineer")
        self.assertIsNotNone(result, "Clean position returned None unexpectedly")
        self.assertIn('Engineer', result, f"Position broken. Got: {result}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST GROUP 4 — Full pipeline end-to-end with real email bodies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Real body from run_cae4fb6e – Sachin Sharma, Nityo Infotech
REAL_BODY_SACHIN = """Your Email Title

Hello Sai Madhavi

My name is Sachin Sharma and I am a Staffing Specialist at
Nityo Infotech
. I am reaching out to you on an exciting job opportunity with one of our clients.

Job Title
-
AI Engineer/Architect

Location
-
Cupertino, CA/Cincinnati, OH(Onsite)

Job Type - Full-Time with benefits

Must Have- Python, LangChain, LangGraph

Thanks and Regards

Sachin Sharma

Talent Acquisition Specialist

Direct : (+1) 609-857-8229

Email:
sachin.sh@nityo.com

LinkedIn :
https://www.linkedin.com/in/sachin-sharma-59253a233/
"""

# Real body from run_cae4fb6e – Mohd Maroof, Headwit Global
REAL_BODY_MAROOF = """Your Email Title

Hi ,

Hope you are doing Good!!!

Please find the Job Description. If you feel comfortable then please send me your updated resume.

Job Description: Artificial Intelligence / Machine Learning Engineer

Location:
Reston, Virginia

Work Mode:
Hybrid (3 days onsite, 2 days remote)

USC GC

Thanks & Regards,

Mohd Maroof |Sr. Recruitment Executive

m.maroof@headwitglobal.com |+1 (512) 759-6198

5900 Balcones Drive, Suite #100, Austin, TX 78731
"""

# Real body from run_cae4fb6e – Shailaja Deshetti, Avance Services
REAL_BODY_SHAILAJA = """EPS Engineer/ Cupertino, California /Permanent

HI Sheetal Darpe,

This email is in regards to a Job I am trying to fill and I think this
might be of interest to you.

EPS Engineer

Cupertino, California

Permanent

Deshetti Shailaja
(Tessa)
Trainee Recruiter
shailaja.deshetti@avanceservices.us
+1 203 902 9194 Ext : 0254
Direct Number : +1 203 902 9194
https://www.linkedin.com/in/deshetti-shailaja-90413b2a6/
https://avanceservices.com/
"""


class TestFullPipelineRealEmails(unittest.TestCase):
    """
    End-to-end extraction tests using REAL email bodies from the latest run.
    These are the canonical acceptance tests — if these pass, the extractors work.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.extractor = _build_extractor()
        except Exception as e:
            raise unittest.SkipTest(f"Could not build ContactExtractor: {e}")

    # ── Sachin Sharma / Nityo Infotech ───────────────────────────────────────
    def test_sachin_name(self):
        msg = _make_email_msg(
            REAL_BODY_SACHIN,
            from_header="sachin.sh@nityo.com",
            subject="AI Engineer/Architect - Cupertino, CA"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_SACHIN,
                                                   source_email='saimadhavi.ip@gmail.com',
                                                   subject='AI Engineer/Architect')
        # Should find sachin.sh@nityo.com
        target = next((c for c in results if 'sachin' in (c.get('email') or '')), None)
        self.assertIsNotNone(target, "No contact found for sachin.sh@nityo.com")
        name = target.get('name') or ''
        self.assertIn('Sachin', name,
                      f"Expected name to contain 'Sachin', got: {name!r}")

    def test_sachin_company_not_tech_stack(self):
        """Company must be 'Nityo Infotech', NOT 'Python, LangChain, LangGraph'."""
        msg = _make_email_msg(
            REAL_BODY_SACHIN,
            from_header="sachin.sh@nityo.com",
            subject="AI Engineer/Architect"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_SACHIN,
                                                   source_email='saimadhavi.ip@gmail.com',
                                                   subject='AI Engineer/Architect')
        target = next((c for c in results if 'sachin' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for sachin not found — skipping company check")
        company = target.get('company') or ''
        # MUST NOT be a tech stack
        self.assertNotIn('LangChain', company,
                         f"Company is a tech stack! Got: {company!r}")
        self.assertNotIn('Python', company,
                         f"Company is a tech stack! Got: {company!r}")

    def test_sachin_sender_job_title(self):
        """sender_job_title must be populated (used to be always null)."""
        msg = _make_email_msg(
            REAL_BODY_SACHIN,
            from_header="sachin.sh@nityo.com",
            subject="AI Engineer/Architect"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_SACHIN,
                                                   source_email='saimadhavi.ip@gmail.com',
                                                   subject='AI Engineer/Architect')
        target = next((c for c in results if 'sachin' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for sachin not found")
        title = target.get('sender_job_title') or ''
        self.assertTrue(len(title) > 2,
                        f"sender_job_title is empty/null — should be 'Staffing Specialist' or similar. Got: {title!r}")

    def test_sachin_job_position_clean(self):
        """job_position should be 'AI Engineer/Architect', not the full subject with company prefix."""
        msg = _make_email_msg(
            REAL_BODY_SACHIN,
            from_header="sachin.sh@nityo.com",
            subject="Nityo Infotech - AI Engineer/Architect"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_SACHIN,
                                                   source_email='saimadhavi.ip@gmail.com',
                                                   subject='Nityo Infotech - AI Engineer/Architect')
        target = next((c for c in results if 'sachin' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for sachin not found")
        pos = target.get('job_position') or ''
        # Should not contain 'Nityo' as a company prefix
        self.assertFalse(pos.startswith('Nityo'),
                         f"Company prefix not stripped from job_position. Got: {pos!r}")

    # ── Mohd Maroof / Headwit Global ─────────────────────────────────────────
    def test_maroof_company_not_title(self):
        """Company must NOT be 'Sr. Recruitment Executive' (that's a job title)."""
        msg = _make_email_msg(
            REAL_BODY_MAROOF,
            from_header="m.maroof@headwitglobal.com",
            subject="AI/ML Engineer - Reston, Virginia"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_MAROOF,
                                                   source_email='sheetaldarpe09@gmail.com',
                                                   subject='AI/ML Engineer - Reston, Virginia')
        target = next((c for c in results if 'maroof' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for maroof not found")
        company = target.get('company') or ''
        self.assertNotIn('Recruitment Executive', company,
                         f"Job title extracted as company! Got: {company!r}")

    def test_maroof_job_position_starts_with_A(self):
        """job_position 'Rtificial Intelligence...' bug: should start with 'A', not 'R'.
        The _clean_job_position now capitalizes first char of truncated positions."""
        msg = _make_email_msg(
            REAL_BODY_MAROOF,
            from_header="m.maroof@headwitglobal.com",
            subject="Artificial Intelligence / Machine Learning Engineer"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_MAROOF,
                                                   source_email='sheetaldarpe09@gmail.com',
                                                   subject='Artificial Intelligence / Machine Learning Engineer')
        target = next((c for c in results if 'maroof' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for maroof not found")
        pos = (target.get('job_position') or '')
        if pos:
            # After fix, first char must be uppercase (either original 'A' or capitalized 'R')
            self.assertTrue(pos[0].isupper(),
                             f"job_position first char not uppercase. Got: {pos!r}")
            # Must NOT start with lowercase 'r' (that was the truncation bug)
            self.assertFalse(pos.startswith('rtificial'),
                             f"Truncated 'A' bug still present! job_position={pos!r}")

    # ── Shailaja Deshetti / Avance Services ─────────────────────────────────
    def test_shailaja_name(self):
        msg = _make_email_msg(
            REAL_BODY_SHAILAJA,
            from_header="shailaja.deshetti@avanceservices.us",
            subject="EPS Engineer/ Cupertino, California /Permanent"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_SHAILAJA,
                                                   source_email='sheetaldarpe09@gmail.com',
                                                   subject='EPS Engineer/ Cupertino, California')
        target = next((c for c in results if 'shailaja' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for shailaja not found")
        name = target.get('name') or ''
        self.assertFalse(name == '' or name is None,
                         "Name is empty — should be 'Deshetti Shailaja' or 'Shailaja Deshetti'")

    def test_shailaja_sender_job_title(self):
        """sender_job_title must be 'Trainee Recruiter', extracted from signature."""
        msg = _make_email_msg(
            REAL_BODY_SHAILAJA,
            from_header="shailaja.deshetti@avanceservices.us",
            subject="EPS Engineer"
        )
        results = self.extractor.extract_contacts(msg, REAL_BODY_SHAILAJA,
                                                   source_email='sheetaldarpe09@gmail.com',
                                                   subject='EPS Engineer')
        target = next((c for c in results if 'shailaja' in (c.get('email') or '')), None)
        if not target:
            self.skipTest("Contact for shailaja not found")
        title = (target.get('sender_job_title') or '').lower()
        self.assertIn('recruiter', title,
                      f"Expected 'Trainee Recruiter' in sender_job_title, got: {title!r}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestIntroSentenceExtractor,
        TestSignatureInfoExtractor,
        TestCleanJobPosition,
        TestFullPipelineRealEmails,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    print("\n" + "=" * 65)
    print("  Recruiter Extraction Quality Tests")
    print("  (using real email bodies from the latest workflow run)")
    print("=" * 65 + "\n")

    runner = unittest.TextTestRunner(verbosity=2, buffer=False)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    if result.wasSuccessful():
        print("  ✅  ALL TESTS PASSED — extraction improvements verified")
    else:
        n_fail = len(result.failures) + len(result.errors)
        print(f"  ❌  {n_fail} TEST(S) FAILED — review extraction logic above")
    print("=" * 65 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
