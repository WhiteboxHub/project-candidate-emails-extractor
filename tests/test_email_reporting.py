"""
Tests for EmailReporter.

Unit tests:
    python -m pytest tests/test_email_reporting.py -v

Live test (sends a real email using the latest report JSON):
    python tests/test_email_reporting.py
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from extractor.reporting.email_reporter import EmailReporter


# ─── Shared test report fixture ───────────────────────────────────────────────

SAMPLE_REPORT = {
    "run_metadata": {
        "run_id": "test-run-001",
        "workflow_id": 2,
        "workflow_key": "email_extractor",
        "started_at": "2026-02-18T09:00:00",
        "finished_at": "2026-02-18T11:30:00",
        "duration_seconds": 9000,
    },
    "summary": {
        "total_candidates": 10,
        "successful_candidates": 7,
        "failed_candidates": 3,
        "total_emails_fetched": 5000,
        "total_emails_inserted": 42,
        "total_duplicates": 800,
        "total_non_vendor": 3500,
    },
    "candidates": [],
    "failed_candidate_details": [
        {
            "candidate_id": 1,
            "candidate_name": "Alice Smith",
            "candidate_email": "alice@example.com",
            "error": "Authentication failed - unable to connect to IMAP server",
        },
        {
            "candidate_id": 2,
            "candidate_name": "Bob Jones",
            "candidate_email": "bob@example.com",
            "error": "Connection timeout after 30s",
        },
        {
            "candidate_id": 3,
            "candidate_name": "Carol White",
            "candidate_email": "carol@example.com",
            "error": "Authentication failed - unable to connect to IMAP server",
        },
    ],
}


# ─── Unit tests ───────────────────────────────────────────────────────────────

class TestEmailReporterDisabled(unittest.TestCase):
    """Reporter with incomplete config should be disabled and not crash."""

    def test_disabled_when_config_incomplete(self):
        reporter = EmailReporter({})
        self.assertFalse(reporter.enabled)

    def test_send_report_does_nothing_when_disabled(self):
        reporter = EmailReporter({})
        # Should not raise
        reporter.send_report(SAMPLE_REPORT)


class TestEmailReporterEnabled(unittest.TestCase):
    """Reporter with full config should send email via SMTP."""

    def setUp(self):
        self.config = {
            "SMTP_SERVER": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USERNAME": "test@test.com",
            "SMTP_PASSWORD": "password",
            "REPORT_FROM_EMAIL": "sender@test.com",
            "REPORT_TO_EMAIL": "receiver@test.com",
        }
        self.reporter = EmailReporter(self.config)

    def test_enabled_with_full_config(self):
        self.assertTrue(self.reporter.enabled)

    @patch("smtplib.SMTP")
    def test_send_report_calls_smtp(self, mock_smtp):
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server

        self.reporter.send_report(SAMPLE_REPORT)

        mock_smtp.assert_called_with("smtp.test.com", 587)
        server.starttls.assert_called_once()
        server.login.assert_called_with("test@test.com", "password")
        server.send_message.assert_called_once()

    @patch("smtplib.SMTP")
    def test_email_headers(self, mock_smtp):
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server

        self.reporter.send_report(SAMPLE_REPORT)

        args, _ = server.send_message.call_args
        msg = args[0]

        self.assertEqual(msg["From"], "sender@test.com")
        self.assertEqual(msg["To"], "receiver@test.com")
        self.assertIn("WBL", msg["Subject"])
        self.assertIn("2026", msg["Subject"])

    @patch("smtplib.SMTP")
    def test_html_body_contains_summary_stats(self, mock_smtp):
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server

        self.reporter.send_report(SAMPLE_REPORT)

        args, _ = server.send_message.call_args
        msg = args[0]
        html = msg.get_payload(0).get_payload()

        self.assertIn("10", html)   # total_candidates
        self.assertIn("7", html)    # successful_candidates
        self.assertIn("3", html)    # failed_candidates
        self.assertIn("42", html)   # total_emails_inserted

    @patch("smtplib.SMTP")
    def test_html_body_contains_all_failed_candidates(self, mock_smtp):
        server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = server

        self.reporter.send_report(SAMPLE_REPORT)

        args, _ = server.send_message.call_args
        msg = args[0]
        html = msg.get_payload(0).get_payload()

        # All 3 failed candidates should appear (no artificial limit anymore)
        self.assertIn("Alice Smith", html)
        self.assertIn("alice@example.com", html)
        self.assertIn("Bob Jones", html)
        self.assertIn("bob@example.com", html)
        self.assertIn("Carol White", html)
        self.assertIn("carol@example.com", html)
        self.assertIn("Authentication failed", html)
        self.assertIn("Connection timeout", html)

    @patch("smtplib.SMTP")
    def test_smtp_error_does_not_raise(self, mock_smtp):
        mock_smtp.side_effect = Exception("SMTP connection refused")
        # Should log error but not propagate
        self.reporter.send_report(SAMPLE_REPORT)


# ─── Live test (run directly) ─────────────────────────────────────────────────

def run_live_test():
    """
    Sends a real email using the latest extraction report JSON.
    Loads SMTP credentials from the project .env file.

    Run with:
        python tests/test_email_reporting.py
    """
    from pathlib import Path
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path)
        print(f"✅ Loaded .env from {dotenv_path}")
    else:
        print(f"⚠️  No .env found at {dotenv_path}")

    config = {
        "SMTP_SERVER": os.getenv("SMTP_SERVER"),
        "SMTP_PORT": os.getenv("SMTP_PORT", "587"),
        "SMTP_USERNAME": os.getenv("SMTP_USERNAME"),
        "SMTP_PASSWORD": os.getenv("SMTP_PASSWORD"),
        "REPORT_FROM_EMAIL": os.getenv("REPORT_FROM_EMAIL"),
        "REPORT_TO_EMAIL": os.getenv("REPORT_TO_EMAIL"),
    }

    reporter = EmailReporter(config)

    if not reporter.enabled:
        print("❌ Email reporter is disabled. Check your .env file.")
        print("   Required: SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, REPORT_FROM_EMAIL, REPORT_TO_EMAIL")
        return

    # Try latest report first, fall back to sample data
    report_path = project_root / "output" / "reports" / "latest_extraction_report.json"

    if report_path.exists():
        print(f"📂 Using latest report: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        print("📂 No latest report found — using sample test data")
        report = SAMPLE_REPORT

    run_id = report.get("run_metadata", {}).get("run_id", "N/A")
    total = report.get("summary", {}).get("total_candidates", 0)
    failed = report.get("summary", {}).get("failed_candidates", 0)

    print(f"📧 Sending report for run {run_id} ({total} candidates, {failed} failed)...")
    reporter.send_report(report)
    print(f"✅ Email sent to {reporter.to_email}")


if __name__ == "__main__":
    run_live_test()
