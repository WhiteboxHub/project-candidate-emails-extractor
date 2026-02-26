import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from extractor.extraction.classification import RecruiterClassifier

def test_classifier():
    classifier = RecruiterClassifier()
    
    test_cases = [
        ("Senior Recruiter", True),
        ("Talent Acquisition Manager", True),
        ("Staffing Specialist", True),
        ("Account Manager", True), # Moderate -> True
        ("HR Director", True),
        ("Software Engineer", False),
        ("Python Developer", False),
        ("Marketing Manager", False),
        (None, False), # No title
    ]
    
    print("Testing RecruiterClassifier with Job Titles:")
    print("-" * 50)
    
    passed = 0
    for title, expected in test_cases:
        is_recruiter, score, reason = classifier.is_recruiter(title)
        result = "PASS" if is_recruiter == expected else "FAIL"
        if result == "PASS":
            passed += 1
        print(f"[{result}] Title: '{title}' -> IsRecruiter: {is_recruiter} (Score: {score}, Reason: {reason})")
        
    print("-" * 50)
    
    # Test Context
    context_cases = [
        ("I am a recruiter at Google", True),
        ("We are looking to hire a developer", True),
        ("Here is the code for the project", False),
    ]
    
    print("\nTesting RecruiterClassifier with Context:")
    print("-" * 50)
    for context, expected in context_cases:
        is_recruiter, score, reason = classifier.is_recruiter(None, context)
        result = "PASS" if is_recruiter == expected else "FAIL"
        if result == "PASS":
            passed += 1
        print(f"[{result}] Context: '{context}' -> IsRecruiter: {is_recruiter} (Score: {score}, Reason: {reason})")

    print(f"\nTotal Passed: {passed}/{len(test_cases) + len(context_cases)}")

if __name__ == "__main__":
    test_classifier()
