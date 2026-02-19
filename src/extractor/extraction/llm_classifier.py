import logging
import os
import json
import re
import httpx
from typing import Dict, List, Optional
import time 

logger = logging.getLogger(__name__)

class LLMJobClassifier:
    """
    Classifier using a Local LLM (via Ollama/FastAPI) to validate job positions.
    Uses generative prompting to provide reasoning and labels.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000", 
        threshold: float = 0.7
    ):
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.base_url = base_url.rstrip('/')
        self.generate_endpoint = f"{self.base_url}/generate" # Removed trailing slash to avoid 307 redirect
        
        # Specific Fix: Use a persistent httpx client for efficiency and reliability
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
            headers={"Content-Type": "application/json"},
            follow_redirects=True
        )
        
        self.logger.info(f"Local LLM initialized at: {self.generate_endpoint}")

    def build_system_prompt(self) -> str:
        """
        Constructs a strict system instruction for JSON-only classification.
        """
        return (
            "You are a strict JSON generator and expert recruitment assistant. "
            "Your task is to classify text into ONE category: 'valid_job' or 'junk'.\n\n"
            "DEFINITIONS:\n"
            "valid_job:\n"
            "- Describes a specific job opening with title, responsibilities, or requirements.\n"
            "junk:\n"
            "- Generic hiring announcements, resumes, signatures, marketing, or newsletters.\n\n"
            "OUTPUT FORMAT (Strict JSON only):\n"
            "{\n"
            "  \"reasoning\": \"One sentence explanation\",\n"
            "  \"label\": \"valid_job\" or \"junk\",\n"
            "  \"confidence\": number between 0.0 and 1.0\n"
            "}"
        )

    def classify(self, text: str) -> Dict:
        """
        Perform local LLM-based classification using prompt-based payload and retry logic.
        """
        if not text:
            return {'label': 'junk', 'score': 0.0, 'is_valid': False, 'reasoning': 'Empty text'}

        # Pre-filter for very short text
        if len(text.split()) < 5:
             return {'label': 'junk', 'score': 1.0, 'is_valid': False, 'reasoning': 'Text too short'}

        max_retries = 3
        backoff = 2

        for attempt in range(max_retries):
            try:
                # Optimized Fix: Use 'prompt' directly as expected by the local server
                # Combine system prompt and user text into one block
                combined_prompt = f"{self.build_system_prompt()}\n\nClassify this job text:\n\n{text[:4000]}"
                
                payload = {
                    "prompt": combined_prompt,
                    "model": "qwen2.5:1.5b",
                    "temperature": 0.0
                }

                self.logger.info(f"  [LLM] Requesting classification (Attempt {attempt + 1})...")
                response = self.client.post("/generate", json=payload)
                
                # Handle rate limiting or server errors with backoff
                if response.status_code in [429, 500, 502, 503, 504]:
                    wait_time = backoff ** (attempt + 1)
                    self.logger.warning(f"  [LLM] API error ({response.status_code}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()
                
                # Check for various response structures
                if isinstance(data, str):
                    output_text = data.strip()
                elif isinstance(data, dict):
                    # Check keys in order of likelihood for this specific server
                    output_text = (
                        data.get('output') or 
                        data.get('response') or 
                        data.get('text') or 
                        data.get('generated_text') or 
                        ""
                    ).strip()
                    
                    # Special check for OpenAI choices format
                    if not output_text and 'choices' in data:
                        output_text = data['choices'][0].get('message', {}).get('content', '').strip()
                else:
                    output_text = str(data).strip()
                
                if not output_text:
                    raise ValueError(f"Empty or unparseable response from LLM. Data: {data}")
                
                # Attempt to parse JSON from response
                result = self._parse_json_from_text(output_text)
                
                label = result.get('label', 'junk').lower()
                score = float(result.get('confidence', 0.5))
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                is_valid = (label == 'valid_job') and (score >= self.threshold)
                
                # Log a summary in a clean format without symbols
                status_label = "VALID" if is_valid else "JUNK"
                self.logger.info(f"  [LLM] Result: {status_label} (score: {score:.2f})")
                self.logger.info(f"  [LLM] Logic : {reasoning}")

                return {
                    'label': "valid" if is_valid else "junk",
                    'score': score,
                    'is_valid': is_valid,
                    'reasoning': reasoning,
                    'raw_llm_output': output_text
                }

            except (httpx.RequestError, ValueError) as e:
                self.logger.error(f"  [LLM] Connection or Parsing Error: {e}")
                if attempt == max_retries - 1:
                    return {'label': 'error', 'score': 0.0, 'is_valid': False, 'reasoning': str(e)}
                time.sleep(backoff ** attempt)

        return {'label': 'error', 'score': 0.0, 'is_valid': False, 'reasoning': 'Unknown error'}

    def _parse_json_from_text(self, text: str) -> Dict:
        """
        Helper to extract JSON from text output if the model was chatty.
        Handles markdown code blocks, extra text, and malformed responses.
        """
        try:
            # First, try direct JSON parse
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON object in the text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Log the actual response for debugging
        self.logger.warning(f"Failed to parse JSON. LLM returned: {text[:500]}")
        
        # Fallback: try to extract label from text
        if "valid_job" in text.lower() and "junk" not in text.lower()[:50]:
            return {'label': 'valid_job', 'confidence': 0.8, 'reasoning': 'Extracted from non-JSON text'}
        
        return {'label': 'junk', 'confidence': 0.8, 'reasoning': 'Failed to parse JSON'}

    def batch_classify(self, texts: List[str]) -> List[Dict]:
        return [self.classify(t) for t in texts]
