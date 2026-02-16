import logging
from typing import Dict, List, Optional
from .database import get_db_client

logger = logging.getLogger(__name__)


def validate_credentials_sql(sql: str, max_test_rows: int = 5) -> Dict:
    """
    Validate that a credentials SQL query returns required columns.
    
    Args:
        sql: SQL query to validate
        max_test_rows: Maximum rows to fetch for testing
        
    Returns:
        Dict with validation results:
        {
            "valid": bool,
            "error": Optional[str],
            "columns_found": List[str],
            "missing_columns": List[str],
            "sample_rows": List[Dict]
        }
    """
    db = get_db_client()
    
    # Add LIMIT to avoid fetching too many rows
    test_sql = sql.strip()
    if "LIMIT" not in test_sql.upper():
        test_sql = f"{test_sql} LIMIT {max_test_rows}"
    
    try:
        results = db.execute_query(test_sql)
        
        if not results:
            return {
                "valid": False,
                "error": "Query returned no rows - check WHERE clause",
                "columns_found": [],
                "missing_columns": [],
                "sample_rows": []
            }
        
        # Check for required column aliases (flexible mapping)
        # DatabaseCandidateSource tries these aliases in order
        required_mappings = {
            "candidate_id": ["candidate_id", "id", "candidate_marketing_id", "candidateMarketingId"],
            "email": ["email", "imap_email", "candidate_email", "username"],
            "imap_password": ["imap_password", "password", "app_password", "email_password"]
        }
        
        # Get all columns from first row
        first_row = results[0]
        columns_found = list(first_row.keys())
        
        # Check each required field
        missing_fields = []
        found_mappings = {}
        
        for field_name, possible_cols in required_mappings.items():
            found = False
            for col in possible_cols:
                if col in first_row and first_row.get(col) not in (None, ""):
                    found_mappings[field_name] = col
                    found = True
                    break
            
            if not found:
                missing_fields.append(field_name)
        
        # Validate data in sample rows
        sample_issues = []
        for idx, row in enumerate(results[:3], 1):
            # Check email format
            email_val = None
            for col in required_mappings["email"]:
                if col in row:
                    email_val = row.get(col)
                    break
            
            if email_val and "@" not in email_val:
                sample_issues.append(f"Row {idx}: Invalid email format '{email_val}'")
            
            # Check password exists
            password_val = None
            for col in required_mappings["imap_password"]:
                if col in row:
                    password_val = row.get(col)
                    break
            
            if not password_val:
                sample_issues.append(f"Row {idx}: Missing IMAP password")
        
        if missing_fields:
            return {
                "valid": False,
                "error": f"Missing required columns: {', '.join(missing_fields)}",
                "columns_found": columns_found,
                "missing_columns": missing_fields,
                "sample_rows": results[:3],
                "found_mappings": found_mappings
            }
        
        if sample_issues:
            return {
                "valid": False,
                "error": f"Data validation issues: {'; '.join(sample_issues)}",
                "columns_found": columns_found,
                "missing_columns": [],
                "sample_rows": results[:3],
                "found_mappings": found_mappings
            }
        
        return {
            "valid": True,
            "error": None,
            "columns_found": columns_found,
            "missing_columns": [],
            "sample_rows": results[:3],
            "found_mappings": found_mappings,
            "total_rows": len(results)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"SQL execution error: {str(e)}",
            "columns_found": [],
            "missing_columns": [],
            "sample_rows": []
        }


def print_validation_report(validation_result: Dict):
    """Print a formatted validation report to console."""
    print("=" * 80)
    print("SQL QUERY VALIDATION REPORT")
    print("=" * 80)
    
    if validation_result["valid"]:
        print("✓ VALIDATION PASSED")
        print()
        print(f"Total test rows: {validation_result.get('total_rows', 'N/A')}")
        print()
        print("Column Mappings:")
        for field, col in validation_result.get("found_mappings", {}).items():
            print(f"  {field:15} → {col}")
        print()
        print("Sample Data (first 3 rows):")
        for idx, row in enumerate(validation_result.get("sample_rows", []), 1):
            print(f"\n  Row {idx}:")
            for key, value in row.items():
                # Mask password
                if "password" in key.lower():
                    display_value = "***" if value else "NULL"
                else:
                    display_value = str(value)[:50]
                print(f"    {key:20} = {display_value}")
    else:
        print("✗ VALIDATION FAILED")
        print()
        print(f"Error: {validation_result.get('error')}")
        print()
        if validation_result.get("columns_found"):
            print("Columns found in query:")
            for col in validation_result["columns_found"]:
                print(f"  - {col}")
        print()
        if validation_result.get("missing_columns"):
            print("Missing required columns:")
            for col in validation_result["missing_columns"]:
                print(f"  - {col}")
    
    print("=" * 80)
