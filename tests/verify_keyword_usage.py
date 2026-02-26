import sys
import os
import re
import csv
import logging
from typing import Dict, List, Set
from unittest.mock import MagicMock

# Mock spacy and gliner before they are imported by the extractors
import sys
sys.modules['spacy'] = MagicMock()
sys.modules['gliner'] = MagicMock()

# Mock torch if needed (gliner usually uses it)
sys.modules['torch'] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractor.filtering.repository import get_filter_repository
from src.extractor.extraction.contacts import ContactExtractor
from src.extractor.extraction.location import LocationExtractor
from src.extractor.extraction.positions import PositionExtractor
from src.extractor.extraction.nlp_spacy import SpacyNERExtractor
from src.extractor.extraction.patterns import RegexExtractor
from src.extractor.extraction.employment_type import EmploymentTypeExtractor

# Suppress logging for clarity
logging.getLogger().setLevel(logging.ERROR)

def get_csv_categories(csv_path: str) -> Set[str]:
    categories = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['category']:
                    categories.add(row['category'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return categories

def verify_keywords():
    # Use relative path from script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, 'src', 'keywords.csv')
    
    print(f"DEBUG: base_dir = {base_dir}")
    print(f"DEBUG: csv_path = {csv_path}")

    if not os.path.exists(csv_path):
        print(f"❌ ERROR: keywords.csv not found at {csv_path}")
        return

    print(f"\n" + "="*60)
    print(f"   KEYWORD LOADING AND USAGE VERIFICATION REPORT")
    print("="*60)
    
    csv_categories = get_csv_categories(csv_path)
    print(f"\n[Core Data]")
    print(f"Total categories in CSV: {len(csv_categories)}")
    
    repo = get_filter_repository()
    keyword_lists = repo.get_keyword_lists()
    loaded_categories = set(keyword_lists.keys())
    
    print(f"Total categories loaded into memory: {len(loaded_categories)}")
    print(f"Loaded categories: {sorted(list(loaded_categories))[:10]} ...")
    
    # Check for missing categories in FilterRepository
    missing_in_repo = csv_categories - loaded_categories
    if missing_in_repo:
        print(f"ERROR: Categories in CSV but NOT loaded: {missing_in_repo}")
    else:
        print(f"OK: All CSV categories loaded into FilterRepository")

    # Initialize Extractors
    print(f"\n[Component Review]")
    config = {'extraction': {'enabled_methods': ['regex', 'spacy']}}
    contact_ext = ContactExtractor(config)
    loc_ext = LocationExtractor()
    pos_ext = PositionExtractor()
    spacy_ext = SpacyNERExtractor()
    reg_ext = RegexExtractor()
    emp_ext = EmploymentTypeExtractor()

    # Track usage
    loaded_by_components = set()

    # CONTACT EXTRACTOR
    print(f"\n--- ContactExtractor ---")
    c_fields = {
        'greeting_patterns': contact_ext.greeting_patterns,
        'company_indicators': contact_ext.company_indicators,
        'skip_header_keywords': contact_ext.skip_keywords
    }
    for name, data in c_fields.items():
        if data:
            print(f"OK: {name:28} : {len(data)} items")
            loaded_by_components.add(name)
        else:
            print(f"ERROR: {name:28} : EMPTY")

    # LOCATION EXTRACTOR
    print(f"\n--- LocationExtractor ---")
    l_fields = {
        'location_false_positives': loc_ext.location_false_positives,
        'us_major_cities': loc_ext.us_major_cities,
        'location_junk_patterns': loc_ext.location_junk_patterns,
        'us_state_abbreviations': loc_ext.us_states,
        'us_state_name_mappings': loc_ext.state_name_to_abbr,
        'location_name_indicators': loc_ext.street_name_indicators
    }
    for name, data in l_fields.items():
        if data:
            print(f"OK: {name:28} : {len(data)} items")
            loaded_by_components.add(name)
        else:
            print(f"ERROR: {name:28} : EMPTY")

    # POSITION EXTRACTOR
    print(f"\n--- PositionExtractor ---")
    p_fields = {
        'position_marketing_words': pos_ext.marketing_words,
        'position_prefixes_remove': pos_ext.prefixes_to_remove,
        'position_trailing_artifacts': pos_ext.trailing_artifacts,
        'html_tag_patterns': pos_ext.html_patterns,
        'job_title_suffixes': pos_ext.job_title_suffixes,
        'acronym_capitalizations': pos_ext.acronym_capitalizations,
        'position_junk_intro_phrases': pos_ext.junk_intro_phrases,
        'blocked_recruiter_titles': pos_ext.recruiter_titles,
        'position_company_prefixes': pos_ext.company_prefixes,
        'position_core_keywords': pos_ext.core_keywords,
        'position_marketing_fluff': pos_ext.marketing_fluff
    }
    for name, data in p_fields.items():
        if data:
            print(f"OK: {name:28} : {len(data)} items")
            loaded_by_components.add(name)
        else:
            print(f"ERROR: {name:28} : EMPTY")

    # SPACY NER EXTRACTOR
    print(f"\n--- SpacyNERExtractor ---")
    s_fields = {
        'job_title_keywords': spacy_ext.job_title_keywords,
        'company_suffix_mapping': spacy_ext.company_suffixes,
        'blocked_ats_domain': spacy_ext.ats_domains,
        'client_language_keywords': spacy_ext.client_keywords,
        'generic_company_terms': spacy_ext.generic_terms,
        'vendor_indicators': spacy_ext.vendor_indicators
    }
    for name, data in s_fields.items():
        if data:
            print(f"OK: {name:28} : {len(data)} items")
            loaded_by_components.add(name)
        else:
            print(f"ERROR: {name:28} : EMPTY")

    # REGEX EXTRACTOR
    print(f"\n--- RegexExtractor ---")
    if reg_ext.blacklist_prefixes:
        print(f"OK: blocked_prefixes/automated   : {len(reg_ext.blacklist_prefixes)} items")
        loaded_by_components.add('blocked_automated_prefix')
        loaded_by_components.add('blocked_generic_prefix')
    else: print(f"ERROR: blocked_prefixes : EMPTY")
    
    if reg_ext.file_extensions:
        print(f"OK: blocked_file_extension      : {len(reg_ext.file_extensions)} items")
        loaded_by_components.add('blocked_file_extension')
    else: print(f"ERROR: blocked_file_extension : EMPTY")

    # EMPLOYMENT TYPE EXTRACTOR
    print(f"\n--- EmploymentTypeExtractor ---")
    # Check if loaded from CSV (Check if property exists or if it's default)
    # We know it's hardcoded currently
    print("ERROR: NOT USING CSV! Currently using hardcoded dictionary in __init__")
    print("   Missing keywords from CSV: 'employment_patterns', 'employment_type_keywords'")

    # GAP ANALYSIS
    print(f"\n" + "="*60)
    print(f"   GAP ANALYSIS")
    print("="*60)
    
    # 1. Unused categories in CSV
    direct_repo_usage = {
        'blocked_personal_domain', 'blocked_edu_domain', 'blocked_test_domain', 
        'blocked_marketing_domain', 'blocked_jobboard_domain', 'blocked_linkedin_domain',
        'blocked_saas_domain', 'blocked_spam_domain', 'blocked_internal_domain',
        'blocked_calendar_domain', 'blocked_emailmarketing_domain', 'blocked_sms_gateway',
        'blocked_social_domain', 'blocked_reply_pattern', 'blocked_linkedin_email',
        'blocked_indeed_email', 'blocked_test_email', 'blocked_exchange',
        'recruiter_keywords', 'anti_recruiter_keywords', 'blocked_exact_email',
        'blocked_system_localpart', 'allowed_jobboard_domain', 'allowed_calendar_domain',
        'allowed_staffing_domain', 'blocked_tracking_prefix', 'blocked_digit_density',
        'blocked_random_string', 'blocked_generic_domain', 'blocked_marketing_subdomain',
        'blocked_spam_tld', 'blocked_training_domain', 'blocked_uuid_pattern',
        'blocked_md5_hash', 'blocked_plus_tracking', 'blocked_desk_prefix',
        'blocked_workday_dots', 'blocked_excessive_subdomains', 'blocked_bot_pattern',
        'recruiter_title_strong', 'recruiter_title_moderate', 'recruiter_title_weak',
        'recruiter_title_negative', 'recruiter_context_positive'
    }
    
    unused = loaded_categories - loaded_by_components - direct_repo_usage
    
    # NLP Indicators
    nlp_indicators = {
        'ner_location_indicators', 'ner_common_cities', 'ner_company_suffixes',
        'location_common_phrases', 'location_tech_terms', 'location_verbs_adjectives',
        'location_invalid_prefixes', 'location_business_suffixes', 'location_html_artifacts',
        'location_generic_words', 'location_prefixes_to_remove', 'employment_patterns',
        'employment_type_keywords', 'position_generic_tech_terms', 'position_portal_indicators',
        'position_false_positives', 'position_company_suffix'
    }
    
    if unused & nlp_indicators:
        print(f"\nERROR: Categories in CSV but NOT used by any component (Hardcoded instead):")
        for u in sorted(list(unused & nlp_indicators)):
            print(f"  - {u}")
    
    # 2. Bug Detection
    print(f"\n--- Logical Bug Detection ---")
    
    # Check LocationExtractor.parse_location_components for state_abbreviations usage
    try:
        import inspect
        source = inspect.getsource(loc_ext.parse_location_components)
        if 'self.state_abbreviations' in source:
            print("ERROR: BUG FOUND: LocationExtractor.parse_location_components references undefined 'self.state_abbreviations'")
        else:
            print("OK: LocationExtractor.parse_location_components bug check passed (or already fixed)")
    except Exception as e:
        print(f"WARN: Could not check LocationExtractor source: {e}")

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_keywords()
