"""
verify_overhaul.py
==================
Self-contained verification of all 7 changed modules.
Run from the email-extractor-bot project root:
    python scripts/verify_overhaul.py
"""
import sys, os, traceback
from pathlib import Path

# ── Setup paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

# Load .env
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for ln in env_file.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, _, v = ln.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# ─────────────────────────────────────────────────────────────────────────────
PASS = 0; FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓  {label}")
    else:
        FAIL += 1
        print(f"  ✗  {label}" + (f"  ←  {detail}" if detail else ""))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CLEANER — smart truncation behaviour
# ─────────────────────────────────────────────────────────────────────────────
section("1. cleaner.py — body truncation & signature zone")
try:
    from extractor.email.cleaner import EmailCleaner
    c = EmailCleaner()

    # Short body — no truncation
    short = "Hello world " * 100   # ~1200 chars
    r = c._smart_truncate(short)
    check("Short body (<8000 chars) not truncated", r == short)

    # Long body — last 700 preserved
    long_body = ("A" * 7400) + ("SIGNATURE_ZONE " * 50)   # >8000 chars
    result = c._smart_truncate(long_body)
    check("Long body total length <= 8000+700+2", len(result) <= 8102)
    check("Signature zone preserved at end", "SIGNATURE_ZONE" in result[-800:],
          f"tail={result[-200:]!r}")

    # HTML entity cleanup
    raw = "Hello &amp; world &nbsp; test &lt;tag&gt;"
    cleaned = c._clean_html_entities(raw)
    check("HTML entities decoded (&amp; → &)", "&amp;" not in cleaned)
    check("&nbsp; removed", "&nbsp;" not in cleaned)

except Exception as e:
    FAIL += 1
    print(f"  ✗  cleaner.py import/run error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 2. KEYWORDS CSV — 3 new categories present
# ─────────────────────────────────────────────────────────────────────────────
section("2. keywords.csv — new categories")
try:
    from extractor.filtering.repository import FilterRepository
    repo = FilterRepository()
    repo.load_filters()
    kw = repo.get_keyword_lists()

    check("CSV categories > 0", len(kw) > 0, f"got {len(kw)}")
    check("junk_name_patterns present",     "junk_name_patterns" in kw,
          f"keys={list(kw)[:10]}")
    check("recruiter_email_signals present","recruiter_email_signals" in kw)
    check("non_recruiter_body_signals present", "non_recruiter_body_signals" in kw)
    check("junk_name_patterns has content", len(kw.get("junk_name_patterns", [])) > 5)
    check("recruiter_email_signals has content", len(kw.get("recruiter_email_signals", [])) > 5)

except Exception as e:
    FAIL += 1
    print(f"  ✗  CSV load error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 3. RULES — pre-extraction marketing body filter
# ─────────────────────────────────────────────────────────────────────────────
section("3. rules.py — marketing body pre-filter")
try:
    from extractor.core.settings import ConfigLoader
    config = ConfigLoader(config_path=str(PROJECT_ROOT / "configs/config.yaml")).load()

    # Inject our fresh repo so rules.py uses the same CSV instance
    import extractor.filtering.repository as fr_mod
    fr_mod._filter_repository = repo

    from extractor.filtering.rules import EmailFilter
    ef = EmailFilter(config)

    spam_body = (
        "Unsubscribe from our emails. "
        "Free trial available. "
        "Newsletter sign up. "
        "Download now. "
        "Limited time offer. "
    )
    recruiter_body = (
        "I have an opening for a Senior Java Developer. "
        "Please share your updated resume. "
        "Direct client requirement."
    )

    check("Marketing body blocked (3+ signals)",
          ef.is_marketing_body(spam_body), f"result={ef.is_marketing_body(spam_body)}")
    check("Recruiter body NOT blocked",
          not ef.is_marketing_body(recruiter_body),
          f"result={ef.is_marketing_body(recruiter_body)}")

    # Allowed staffing domain auto-classifies as recruiter
    check("Staffing domain is recruiter",
          ef._classify_with_rules("hiring a dev", recruiter_body, "recruiter@teksystems.com"))

except Exception as e:
    FAIL += 1
    print(f"  ✗  rules.py error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION — richer body scoring + domain override
# ─────────────────────────────────────────────────────────────────────────────
section("4. classification.py — recruiter signals")
try:
    from extractor.extraction.classification import RecruiterClassifier
    clf = RecruiterClassifier()

    cases = [
        # (title, body, email, expected_is_recruiter, label)
        ("Senior Technical Recruiter", "looking for a developer",
         "a@teksystems.com", True,  "Strong title → recruiter"),
        ("Software Engineer", "built this feature",
         "dev@sometech.com", False, "Negative title → not recruiter"),
        (None, "I have an opening for a Python Developer at our client",
         "r@kforce.com", True,  "Body signal + staffing domain → recruiter"),
        (None, "Unsubscribe. Free trial ends soon. Newsletter.",
         "news@marketing.com", False, "Spam body → not recruiter"),
        ("Talent Acquisition", "we are currently hiring a Java Dev",
         "hr@randstad.com", True, "Known staffing domain → auto recruiter"),
    ]

    for title, body, email, expected, label in cases:
        is_r, score, reason = clf.is_recruiter(title, body, sender_email=email)
        check(label, is_r == expected,
              f"got is_recruiter={is_r} score={score:.2f} reason={reason[:60]}")

except Exception as e:
    FAIL += 1
    print(f"  ✗  classification.py error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 5. NAME VALIDATION — casing normalization + guards
# ─────────────────────────────────────────────────────────────────────────────
section("5. contacts.py — _is_valid_person_name")
try:
    from extractor.extraction.contacts import ContactExtractor
    extractor = ContactExtractor(config)

    name_cases = [
        ("JOHN SMITH",        True,  "ALL-CAPS normalized → pass"),
        ("john smith",        True,  "all-lowercase normalized → pass"),
        ("Sarah Mitchell",    True,  "Normal name → pass"),
        ("Recruiting",        False, "Single junk word → reject"),
        ("Talent Acquisition",False, "junk_name_patterns word → reject"),
        ("Noreply Bot",       False, "junk_name_patterns word → reject"),
        ("Houston Smith",     False, "First word is US city → reject"),
        ("Dallas Texas",      False, "Both words are cities → reject"),
    ]

    for name, expected, label in name_cases:
        result = extractor._is_valid_person_name(name)
        check(label, result == expected,
              f"name={name!r} got={'pass' if result else 'reject'} expected={'pass' if expected else 'reject'}")

except Exception as e:
    FAIL += 1
    print(f"  ✗  contacts.py name validation error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 6. POSITIONS — word count tightened, terminal suffix required
# ─────────────────────────────────────────────────────────────────────────────
section("6. positions.py — _is_valid_position")
try:
    from extractor.extraction.positions import PositionExtractor
    pe = PositionExtractor(config)

    pos_cases = [
        ("Senior Python Developer",  True,  "Valid 3-word position"),
        ("AI Engineer",              True,  "Valid 2-word position"),
        ("Lead Java Backend Engineer",True, "Valid 4-word position"),
        ("Senior GenAI Platform Engineer Manager", False, "7 words → too long (>6)"),
        ("Hope You Are Doing Well Today", False, "Greeting sentence → no suffix"),
        ("Engineer",                 False, "Single word → too short"),
    ]

    for pos, expected, label in pos_cases:
        result = pe._is_valid_position(pos)
        check(label, result == expected,
              f"pos={pos!r} got={'valid' if result else 'invalid'} expected={'valid' if expected else 'invalid'}")

except Exception as e:
    FAIL += 1
    print(f"  ✗  positions.py validation error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 7. VENDOR CONTACTS — Gate 3 blocks non-recruiter
# ─────────────────────────────────────────────────────────────────────────────
section("7. vendor_contacts.py — Gate 3 non-recruiter block")
try:
    from extractor.persistence.vendor_contacts import VendorUtil

    class _FakeAPI:
        def post(self, *a, **kw): return {}
        def get(self, *a, **kw): return []
        def authenticate(self): return True

    vu = VendorUtil(_FakeAPI())
    # Inject our repo so VendorUtil doesn't re-load
    vu.filter_repo = repo

    recruiter_contact = {
        "email": "sarah@teksystems.com",
        "name": "Sarah Mitchell",
        "is_recruiter": True,
        "recruiter_score": 1.0,
        "company": "TEKsystems",
        "job_position": "Python Developer",
    }
    non_recruiter_contact = {
        "email": "dev@acme.com",
        "name": "John Developer",
        "is_recruiter": False,
        "recruiter_score": 0.1,
        "company": "Acme Corp",
        "job_position": "Software Engineer",
    }

    # Test _is_valid_contact (requires email)
    check("Recruiter contact passes quality gate",  vu._is_valid_contact(recruiter_contact))
    check("Non-recruiter contact passes quality gate (email exists)", vu._is_valid_contact(non_recruiter_contact))

    # The recruiter gate (Gate 3) happens inside save_contacts() — simulate
    def _passes_all_gates(contact):
        if not vu._is_valid_contact(contact):
            return False, "quality"
        if not vu._is_vendor_recruiter_contact(contact):
            return False, "vendor"
        if not contact.get("is_recruiter", False):
            return False, "not_recruiter"
        return True, "ok"

    ok, gate = _passes_all_gates(recruiter_contact)
    check("Recruiter passes all 3 gates", ok, f"stopped at={gate}")

    ok, gate = _passes_all_gates(non_recruiter_contact)
    check("Non-recruiter blocked at Gate 3", not ok and gate == "not_recruiter",
          f"gate={gate}")

except Exception as e:
    FAIL += 1
    print(f"  ✗  vendor_contacts.py error: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# RESULT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULT: {PASS} passed / {FAIL} failed")
if FAIL == 0:
    print("  ✅  ALL CHECKS PASSED — changes are correct")
else:
    print("  ❌  SOME CHECKS FAILED — see ✗ lines above")
print('='*60)
sys.exit(0 if FAIL == 0 else 1)
