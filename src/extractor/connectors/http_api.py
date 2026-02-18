import httpx
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
import time

logger = logging.getLogger(__name__)

class APIClient:
    """
    API client for Whitebox Learning platform
    Handles authentication and API requests using persistent httpx session
    """
    
    DEFAULT_TIMEOUT = 120
    
    def __init__(self, base_url: str, email: str, password: str, employee_id: int):
        self.base_url = base_url.rstrip('/')
        self.email = email
        self.password = password
        self.employee_id = employee_id
        self.token = None
        self.token_expiry = None
        self.logger = logging.getLogger(__name__)
        
        # specific fix #2: Use persistent session
        self.session = httpx.Client(
            base_url=self.base_url, 
            timeout=self.DEFAULT_TIMEOUT,
            verify=False,  # optimization for local/dev env if needed, usually True for prod
            follow_redirects=True  # Fix for 307 redirects
        )
        # Initialize standard headers
        self.session.headers.update({
            # "Content-Type": "application/json",  <- REMOVE THIS. httpx sets it automatically for json= param.
            # Setting it globally breaks data= param (form-encoded) login, causing 422.
            "X-Employee-ID": str(self.employee_id)
        })

    def _is_token_valid(self) -> bool:
        """Check if current token is still valid (with buffer time)"""
        if not self.token or not self.token_expiry:
            return False
        return datetime.now() < (self.token_expiry - timedelta(minutes=5))

    def authenticate(self) -> bool:
        """
        Authenticate with the API and get bearer token
        Uses OAuth2 form-encoded authentication
        """
        try:
            # Login endpoint - FastAPI OAuth2
            # Note: base_url is already in session, but login often implies full URL or relative
            # httpx client with base_url handles relative paths
            
            form_data = {
                "username": self.email,
                "password": self.password,
                "grant_type": "password"
            }
            
            # Login usually doesn't need auth headers (which might be expired)
            # We can use a separate call or transient client if needed, 
            # but usually posting to /api/login is open.
            
            response = self.session.post("/api/login", data=form_data)
            
            if response.status_code != 200:
                self.logger.error(f"Login failed with status {response.status_code}")
                return False
            
            data = response.json()
            self.token = data.get('access_token')
            
            if not self.token:
                self.logger.error("No access_token in authentication response")
                return False
            
            # Set token expiry
            self.token_expiry = datetime.now() + timedelta(hours=1)
            
            # specific fix #1: Update session headers
            self.session.headers.update({
                "Authorization": f"Bearer {self.token}"
            })
            
            self.logger.info(f"Successfully authenticated as {self.email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False

    def _ensure_auth(self):
        """Ensure valid session auth header exists"""
        if not self._is_token_valid():
            if not self.authenticate():
                raise Exception("Failed to authenticate with API")

    def _handle_request_with_retry(self, method_name, endpoint, **kwargs):
        """
        Execute request with 401 token refresh and 429 backoff
        """
        max_retries = 3
        backoff = 2
        
        # Ensure follow_redirects is enabled for all requests
        if 'follow_redirects' not in kwargs:
            kwargs['follow_redirects'] = True

        for attempt in range(max_retries):
            # Ensure auth before request
            if attempt == 0:
                self._ensure_auth()
            
            try:
                # Construct full URL to avoid any ambiguity with httpx base_url/redirects
                url = f"{self.base_url}{endpoint}" if endpoint.startswith('/') else endpoint
                
                # Use request method directly on session
                # getattr(self.session, method_name) might be bound to relative path logic
                # Let's use self.session.request for maximum control, or mapped method
                method = getattr(self.session, method_name)
                
                if method_name in ['post', 'put', 'patch']:
                    self.logger.info(f"DEBUG: {method_name.upper()} {url} | Payload: {kwargs.get('json', 'No JSON')}")

                # Note: If we use full URL, we should pass it. httpx handles full URL even if base_url is set.
                response = method(url, **kwargs)
                
                # Check for 401 Unauthorized
                if response.status_code == 401:
                    self.logger.warning(f"Request to {endpoint} returned 401. Refreshing token...")
                    if self.authenticate():
                        # Retry
                        continue
                    else:
                        raise Exception("Authentication failed during retry")

                # Check for 429 Too Many Requests
                if response.status_code == 429:
                    wait_time = backoff ** attempt
                    self.logger.warning(f"Rate limited (429) on {endpoint}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                return response
                
            except httpx.RequestError as e:
                self.logger.error(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        
        raise Exception(f"Max retries exceeded for {endpoint}")
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        response = self._handle_request_with_retry('get', endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint: str, data: Dict) -> Any:
        response = self._handle_request_with_retry('post', endpoint, json=data)
        if response.status_code >= 400:
            self.logger.error(f"POST {endpoint} failed: {response.status_code}")
        response.raise_for_status()
        return response.json()
    
    def put(self, endpoint: str, data: Dict) -> Any:
        response = self._handle_request_with_retry('put', endpoint, json=data)
        self.logger.info(f"PUT {endpoint} | Status: {response.status_code}")
        response.raise_for_status()
        return response.json()

    def patch(self, endpoint: str, data: Dict) -> Any:
        response = self._handle_request_with_retry('patch', endpoint, json=data)
        self.logger.info(f"PATCH {endpoint} | Status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    
    def delete(self, endpoint: str) -> Any:
        response = self._handle_request_with_retry('delete', endpoint)
        response.raise_for_status()
        return response.json()

def get_api_client() -> APIClient:
    """Factory function for APIClient"""
    base_url = os.getenv('API_BASE_URL')
    email = os.getenv('API_EMAIL')
    password = os.getenv('API_PASSWORD')
    employee_id = int(os.getenv('EMPLOYEE_ID', 0))
    
    if not all([base_url, email, password, employee_id]):
        raise ValueError("Missing required environment variables")
    
    return APIClient(base_url, email, password, employee_id)
