import sys
import os
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from extractor.extraction.location import LocationExtractor
from extractor.extraction.positions import PositionExtractor
from extractor.extraction.nlp_spacy import SpacyNERExtractor

def verify_robustness():
    print("Starting robustness verification...")
    
    # 1. Vendor Span Extraction (with _ and spaces)
    print("\n--- Testing Vendor Span Extraction ---")
    try:
        spacy_extractor = SpacyNERExtractor()
        
        # Enable logging to console for debug
        import logging
        logging.basicConfig(level=logging.INFO)
        
        html_case1 = '<span>Recruiter - _Acme Corp_</span>'
        res1 = spacy_extractor.extract_vendor_from_span(html_case1)
        print(f"Case 1 (_Acme Corp_): {res1}")
        
        html_case2 = '<span>Hiring Manager: _Tech Results Inc_</span>'
        res2 = spacy_extractor.extract_vendor_from_span(html_case2)
        print(f"Case 2 (_Tech Results Inc_): {res2}")
        
    except Exception as e:
        print(f"Error in Vendor Span test: {e}")

    # 2. Position Extraction (with _ )
    print("\n--- Testing Position Extraction ---")
    try:
        pos_extractor = PositionExtractor()
        
        pos_text1 = "We are looking for a _Senior Java Developer_ to join..."
        pos1 = pos_extractor.extract_job_position_regex(pos_text1)
        print(f"Case 1 (_Senior Java Developer_): {pos1}")

        pos_text2 = "Title: _Lead Architect_"
        pos2 = pos_extractor.extract_job_position_regex(pos_text2)
        print(f"Case 2 (_Lead Architect_): {pos2}")
        
    except Exception as e:
        print(f"Error in Position test: {e}")

    # 3. Location Extraction (with _)
    print("\n--- Testing Location Extraction ---")
    try:
        loc_extractor = LocationExtractor()
        
        loc_text1 = "Location: _Dallas_, TX"
        loc1 = loc_extractor.extract_location_with_zip(loc_text1)
        print(f"Case 1 (_Dallas_, TX): {loc1}")

        loc_text2 = "Based in _San Francisco_, CA"
        loc2 = loc_extractor.extract_location_with_zip(loc_text2)
        print(f"Case 2 (_San Francisco_, CA): {loc2}")
        
    except Exception as e:
        print(f"Error in Location test: {e}")

if __name__ == "__main__":
    verify_robustness()
