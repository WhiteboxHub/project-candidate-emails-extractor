#!/usr/bin/env python3
"""
Verification script to ensure all keywords are loading from keywords.csv
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.filters.filter_repository import get_filter_repository

def verify_csv_loading():
    """Verify that all expected keyword categories are loading from CSV"""
    
    print("=" * 80)
    print("VERIFYING KEYWORDS.CSV LOADING")
    print("=" * 80)
    
    # Get filter repository
    filter_repo = get_filter_repository()
    keyword_lists = filter_repo.get_keyword_lists()
    
    # Expected categories that should be in CSV based on actual content
    expected_categories = {
        # Domain filtering
        'allowed_staffing_domain': 'Legitimate staffing domains',
        'blocked_personal_domain': 'Personal email domains',
        'blocked_edu_domain': 'Educational domains',
        'blocked_test_domain': 'Test domains',
        'blocked_marketing_domain': 'Marketing domains',
        'blocked_jobboard_domain': 'Job board platforms',
        'blocked_linkedin_domain': 'LinkedIn automated domains',
        'blocked_saas_domain': 'SaaS/product companies',
        'blocked_spam_domain': 'Spam/temp mail services',
        'blocked_internal_domain': 'Internal company domains',
        'blocked_calendar_domain': 'Calendar/meeting platforms',
        'blocked_emailmarketing_domain': 'Email marketing services',
        'blocked_ats_domain': 'ATS platform domains',
        'blocked_sms_gateway': 'SMS gateways',
        'blocked_social_domain': 'Social media platforms',
        
        # Localpart / Pattern filtering
        'blocked_automated_prefix': 'Automated email prefixes',
        'blocked_generic_prefix': 'Generic automated prefixes',
        'blocked_reply_pattern': 'Auto-generated reply patterns',
        'blocked_linkedin_email': 'LinkedIn specific patterns',
        'blocked_indeed_email': 'Indeed specific patterns',
        'blocked_test_email': 'Test email patterns',
        'blocked_exchange': 'Exchange system emails',
        'blocked_exact_email': 'Specific blocked addresses',
        'blocked_system_localpart': 'Blocked system localparts',
        'blocked_tracking_prefix': 'Tracking ID prefixes',
        'blocked_digit_density': 'High digit density regex',
        'blocked_random_string': 'Random string regex',
        'blocked_uuid_pattern': 'UUID regex',
        'blocked_md5_hash': 'Hash regex',
        'blocked_plus_tracking': 'Plus-based tracking',
        'blocked_desk_prefix': 'Desk pattern regex',
        'blocked_workday_dots': 'Workday multiple dots',
        'blocked_excessive_subdomains': 'Excessive subdomains',
        'blocked_bot_pattern': 'Generic bot pattern',
        
        # Allower lists
        'allowed_jobboard_domain': 'Allowed recruiter job boards',
        'allowed_calendar_domain': 'Allowed calendar domains',
        'recruiter_keywords': 'Positive recruiter keywords',
        
        # Position extraction
        'job_position_trigger_words': 'Job position trigger words',
        'position_marketing_words': 'Marketing/adjective removals',
        'position_prefixes_remove': 'Prefixes to remove',
        'position_trailing_artifacts': 'Trailing artifacts to remove',
        'job_title_suffixes': 'Job title suffixes',
        'acronym_capitalizations': 'Acronym case mappings',
        'position_junk_intro_phrases': 'Junk intro phrases',
        'blocked_recruiter_titles': 'Recruiter titles to block',
        'position_company_prefixes': 'Company prefixes to remove',
        'position_core_keywords': 'Required core job keywords',
        'position_marketing_fluff': 'Marketing fluff phrases',
        'position_generic_tech_terms': 'Generic core tech terms',
        'position_portal_indicators': 'Portal/system indicators',
        'position_false_positives': 'Final position false positives',
        'position_company_suffix': 'Company suffixes in positions',
        
        # Employment extraction
        'employment_patterns': 'Specialized employment regex mappings',
        
        # Location extraction
        'location_common_phrases': 'Location common phrases',
        'location_tech_terms': 'Location tech terms to ignore',
        'location_verbs_adjectives': 'Location verbs/adjectives to ignore',
        'location_invalid_prefixes': 'Invalid location start words',
        'location_business_suffixes': 'Business suffixes as locations',
        'location_html_artifacts': 'HTML artifacts in locations',
        'location_generic_words': 'Generic words as locations',
        'location_prefixes_to_remove': 'Explicit prefixes to strip',
        'location_name_indicators': 'Street name indicators',
        'location_false_positives': 'Location false positives',
        'us_major_cities': 'Known major US cities',
        'us_state_abbreviations': 'US state abbreviations',
        'us_state_name_mappings': 'State name to abbr mappings',
        
        # NER / Smart Extraction
        'ner_location_indicators': 'NER location indicators',
        'ner_common_cities': 'NER common cities',
        'ner_company_suffixes': 'NER confidence company suffixes',
        'job_title_keywords': 'Job title keywords (NER)',
        'company_suffix_mapping': 'Company suffix standardization',
        'client_language_keywords': 'Client vs Vendor language',
        'generic_company_terms': 'Generic noise company terms',
        'vendor_indicators': 'Vendor company indicators',
        
        # Other
        'anti_recruiter_keywords': 'Negative recruiter keywords',
        'greeting_patterns': 'Greeting patterns to strip',
        'company_indicators': 'Generic company/team indicators',
        'skip_header_keywords': 'System header keywords to skip',
        'html_tag_patterns': 'HTML tag stripping regexes',
    }
    
    print(f"\n✓ Loaded {len(keyword_lists)} categories from CSV\n")
    
    # Check each expected category
    found = 0
    missing = []
    
    for category, description in sorted(expected_categories.items()):
        if category in keyword_lists:
            count = len(keyword_lists[category])
            print(f"✓ {category:40s} - {count:3d} keywords - {description}")
            found += 1
        else:
            print(f"✗ {category:40s} - MISSING - {description}")
            missing.append(category)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {found}/{len(expected_categories)} categories found")
    print("=" * 80)
    
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} categories are missing from CSV:")
        for cat in missing:
            print(f"  - {cat}")
    else:
        print("\n✅ ALL EXPECTED CATEGORIES LOADED SUCCESSFULLY!")
    
    # Test filter_repository.check_email() functionality
    print("\n" + "=" * 80)
    print("TESTING EMAIL FILTERING FUNCTIONALITY")
    print("=" * 80)
    
    test_emails = [
        ('noreply@linkedin.com', 'block', 'Automated LinkedIn'),
        ('info@company.com', 'block', 'Generic info@'),
        ('jane.doe@gmail.com', 'block', 'Personal Gmail'),
        ('jobs@greenhouse.io', 'block', 'ATS domain (greenhouse)'),
        ('d45493db-1629-4a02-affb-11f17d2500f6@reply.linkedin.com', 'block', 'UUID pattern'),
        ('bb2137b38d8f4e81beb7fecf9d1785a6@integrisit.com', 'block', 'MD5 hash'),
        ('recruiter@teksystems.com', 'allow', 'Allowed staffing domain'),
        ('valid.contact@amazon.com', None, 'Valid corporate domain (Amazon)'),
    ]
    
    print("\nTesting email filtering:")
    passed = 0
    failed = 0
    
    for email, expected, description in test_emails:
        result = filter_repo.check_email(email)
        status = "✓" if result == expected else "✗"
        
        if result == expected:
            passed += 1
            print(f"{status} {email:50s} -> {str(result):15s} (expected {str(expected):15s}) - {description}")
        else:
            failed += 1
            print(f"{status} {email:50s} -> {str(result):15s} (expected {str(expected):15s}) - {description} FAILED!")
    
    print(f"\n{passed}/{len(test_emails)} tests passed")
    
    if failed > 0:
        print(f"⚠ {failed} tests failed - check CSV patterns and priority logic")
    else:
        print("✅ ALL EMAIL FILTERING TESTS PASSED!")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    return len(missing) == 0 and failed == 0

if __name__ == '__main__':
    success = verify_csv_loading()
    sys.exit(0 if success else 1)
