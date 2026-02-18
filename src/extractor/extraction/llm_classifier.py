import logging
import os
import json
import re
import httpx
from typing import Dict, List, Optional

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
        self.generate_endpoint = f"{self.base_url}/generate"
        
        self.logger.info(f"Local LLM initialized at: {self.generate_endpoint}")

    def build_system_prompt(self) -> str:
        """
        Constructs a strict system instruction for JSON-only classification.
        """

        return (
            "You are a strict JSON generator and expert recruitment assistant.\n"
            "Your task is to classify the given text into ONE of two categories:\n"
            "- valid_job\n"
            "- junk\n\n"

            "CLASSIFICATION RULES:\n"
            "valid_job:\n"
            "- Must describe a SPECIFIC open position (e.g., 'Senior Python Developer', 'Data Analyst')\n"
            "- Must include specific responsibilities or requirements for that role\n"
            "- NOT a general 'we are hiring' announcement\n\n"

            "junk:\n"
            "- General 'We are hiring' or 'Join our team' posts without specific role details\n"
            "- Lists of multiple potential roles without details (e.g., 'Hiring Java, .NET, QA')\n"
            "- Email signatures\n"
            "- Company advertisements\n"
            "- Candidate profiles looking for jobs\n"
            "- Spam, newsletters, marketing\n\n"

            "CRITICAL OUTPUT RULES:\n"
            "- Output ONLY valid JSON\n"
            "- No explanations outside JSON\n"
            "- No markdown\n"
            "- No extra text\n"
            "- No prefixes or suffixes\n\n"

            "OUTPUT FORMAT:\n"
            "{\n"
            "  \"reasoning\": \"One sentence explanation\",\n"
            "  \"label\": \"valid_job\" or \"junk\",\n"
            "  \"confidence\": number between 0.0 and 1.0\n"
            "}\n\n"

            "OUTPUT JSON:"
        )
    def classify(self, text: str) -> Dict:
        """
        Perform local LLM-based classification.
        """
        if not text:
            return {'label': 'junk', 'confidence': 1.0, 'reasoning': 'Empty text'}

        # Fix #3: Add junk keyword filter before saving/processing
        junk_keywords = [
            "we are hiring", "join our team", "hiring now", "open positions", 
            "click here", "subscribe", "newsletter", "follow us"
        ]
        text_lower = text.lower()
        if len(text.split()) < 10:  # Too short
             return {'label': 'junk', 'confidence': 1.0, 'reasoning': 'Text too short'}
             
        # If text is dominated by junk keywords without specific role details
        # This is a simple heuristic; LLM is better, but this saves tokens
        # We'll let LLM handle the nuance, but if it's JUST "We are hiring" we skip
        if any(text_lower == k for k in junk_keywords):
             return {'label': 'junk', 'confidence': 0.9, 'reasoning': 'Generic hiring slogan'}

        try:
            # Truncate text to keep prompt within reasonable limits
            if len(text) > 4000:
                text = text[:4000]

            payload = {
                "prompt": f"{self.build_system_prompt()}\n\nClassify this text:\n\n{text}",
                "system": self.build_system_prompt() # If the local API supports a system field
            }
            
            # Note: We use a synchronous request here to match the existing BERT/Groq implementation pattern
            # in the orchestrator, but httpx allows for easy async later if needed.
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.generate_endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
            
            # The structure of 'data' depends on the local API server's implementation.
            # Usually, local wrappers return the text in a 'text' or 'response' key.
            output_text = data.get('text', data.get('response', data.get('generated_text', data.get('output', '')))).strip()
            
            # Log the raw response for debugging
            self.logger.info(f"Raw LLM response (first 300 chars): {output_text[:300]}")

            # Attempt to parse JSON from response if the local model didn't return pure JSON
            result = self._parse_json_from_text(output_text)
            
            self.logger.info(f"LLM Classification reasoning for raw job: {result.get('reasoning', '')}")

            label = result.get('label', 'junk').lower()
            score = float(result.get('confidence', 0.5))
            
            is_valid = (label == 'valid_job') and (score >= self.threshold)
            
            return {
                'label': "valid" if is_valid else "junk",
                'score': score,
                'is_valid': is_valid,
                'reasoning': result.get('reasoning', ''),
                'raw_llm_output': output_text
            }
            
        except Exception as e:
            self.logger.error(f"Local LLM Classification error: {e}")
            return {'label': 'error', 'score': 0.0, 'is_valid': False}

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
