class JobLeadExtractorError(Exception):
    """Base exception for JobLead Extractor."""
    pass

class ConfigurationError(JobLeadExtractorError):
    """Raised when there is a configuration error."""
    pass

class ExtractionError(JobLeadExtractorError):
    """Raised when extraction fails."""
    pass
