import requests
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class APIClient:
    """
    API client for Whitebox Learning platform
    Handles authentication and API requests
    """
    
    def __init__(self, base_url: str, email: str, password: str, employee_id: int):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.employee_id = employee_id
        self.token = None
        self.token_expiry = None
        self.logger = logging.getLogger(__name__)
    
    def _is_token_valid(self) -> bool:
        """Check if current token is still valid"""
        if not self.token or not self.token_expiry:
            return False
        return datetime.now() < self.token_expiry
    
    def authenticate(self) -> bool:
        """
        Authenticate with the API and get bearer token
        Uses OAuth2 form-encoded authentication
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Login endpoint - FastAPI OAuth2
            login_url = f"{self.base_url}/api/login"
            
            # OAuth2PasswordRequestForm expects form-encoded data with 'username' and 'password'
            form_data = {
                "username": self.email,
                "password": self.password,
                "grant_type": "password"  # Required by OAuth2 spec
            }
            
            response = requests.post(login_url, data=form_data, timeout=30)
            
            # Log response for debugging
            if response.status_code != 200:
                self.logger.error(f"Login failed with status {response.status_code}")
                self.logger.error(f"Response: {response.text}")
            
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get('access_token')
            
            if not self.token:
                self.logger.error("No access_token in authentication response")
                self.logger.error(f"Response data: {data}")
                return False
            
            # Set token expiry (default 1 hour if not provided)
            # JWT tokens typically expire in 1 hour
            self.token_expiry = datetime.now() + timedelta(hours=1)
            
            self.logger.info(f"Successfully authenticated as {self.email}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during authentication: {str(e)}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token"""
        if not self._is_token_valid():
            if not self.authenticate():
                raise Exception("Failed to authenticate with API")
        
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        Make GET request to API
        
        Args:
            endpoint: API endpoint (e.g., '/candidate/marketing')
            params: Query parameters
            
        Returns:
            Response data
        """
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"GET request failed for {endpoint}: {str(e)}")
            raise
    
    def post(self, endpoint: str, data: Dict) -> Any:
        """
        Make POST request to API
        
        Args:
            endpoint: API endpoint
            data: Request body data
            
        Returns:
            Response data
        """
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            # Log error responses for debugging
            if response.status_code >= 400:
                self.logger.error(f"POST {endpoint} failed with status {response.status_code}")
                self.logger.error(f"Response: {response.text}")
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"POST request failed for {endpoint}: {str(e)}")
            raise
    
    def put(self, endpoint: str, data: Dict) -> Any:
        """
        Make PUT request to API
        
        Args:
            endpoint: API endpoint
            data: Request body data
            
        Returns:
            Response data
        """
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            response = requests.put(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"PUT request failed for {endpoint}: {str(e)}")
            raise
    
    def delete(self, endpoint: str) -> Any:
        """
        Make DELETE request to API
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response data
        """
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self._get_headers()
            
            response = requests.delete(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DELETE request failed for {endpoint}: {str(e)}")
            raise


def get_api_client() -> APIClient:
    """
    Factory function to create API client from environment variables
    
    Returns:
        Configured APIClient instance
    """
    base_url = os.getenv('API_BASE_URL')
    email = os.getenv('API_EMAIL')
    password = os.getenv('API_PASSWORD')
    employee_id = int(os.getenv('EMPLOYEE_ID', 0))
    
    if not all([base_url, email, password, employee_id]):
        raise ValueError(
            "Missing required environment variables: "
            "API_BASE_URL, API_EMAIL, API_PASSWORD, EMPLOYEE_ID"
        )
    
    return APIClient(base_url, email, password, employee_id)
