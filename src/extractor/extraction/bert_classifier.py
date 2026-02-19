import logging
import torch
import os
from transformers import pipeline
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BertJobClassifier:
    """
    Classifier using BERT to validate job positions.
    Defaults to zero-shot classification to handle "Valid Job" vs "Junk" logic.
    """
    
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased", 
        zero_shot_model: str = "valhalla/distilbart-mnli-12-1",
        device: int = -1, 
        threshold: float = 0.5
    ):
        self.logger = logging.getLogger(__name__)
        self.threshold = threshold
        self.model_type = "binary" # Default attempt
        
        # Detect device
        if device == -1 and torch.cuda.is_available():
            self.device = 0
            self.logger.info("CUDA available - using GPU")
        else:
            self.device = -1
            self.logger.info("Using CPU")

        # Check for local fine-tuned model first
        # 1. Try relative to current working directory
        cwd_model_path = os.path.join(os.getcwd(), "models", "bert_binary_classifier")
        # 2. Try relative to project root (4 levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        root_model_path = os.path.join(project_root, "models", "bert_binary_classifier")
        
        selected_model = model_name
        if os.path.exists(cwd_model_path):
            self.logger.info(f"Using local model from CWD: {cwd_model_path}")
            selected_model = cwd_model_path
        elif os.path.exists(root_model_path):
            self.logger.info(f"Using local model from project root: {root_model_path}")
            selected_model = root_model_path
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.logger.info(f"Loading binary classifier components from: {selected_model}")
            
            # Explicitly load tokenizer and model for better control
            tokenizer = AutoTokenizer.from_pretrained(selected_model)
            model = AutoModelForSequenceClassification.from_pretrained(selected_model)
            
            self.classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )
            self.model_type = "binary"
            self.logger.info("✓ Binary classifier (model + tokenizer) loaded successfully")
            
        except Exception as e:
            self.logger.error(f"FAILED to load binary classifier from {selected_model}")
            self.logger.error(f"Error details: {e}")
            import traceback
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            self.logger.info(f"Falling back to Zero-Shot: {zero_shot_model}")
            self.classifier = pipeline(
                "zero-shot-classification",
                model=zero_shot_model,
                device=self.device
            )
            self.model_type = "zero-shot"
            self.candidate_labels = ["valid job requirement", "junk text or spam"]
            self.valid_labels = ["valid job requirement"]
            self.logger.info("✓ Zero-Shot fallback initialized")
        except Exception as e:
            self.logger.error(f"Crirical error initializing BERT classifier: {e}")
            raise

    def classify(self, text: str) -> Dict:
        """
        Perform binary classification on the input text.
        
        Returns:
            Dict with 'label', 'score', 'is_valid'
        """
        if not text:
            return {'label': 'empty', 'score': 0.0, 'is_valid': False}

        try:
            # Handle BERT token limits (approx 512 tokens, ~1000-2000 chars)
            if len(text) > 2000:
                text = text[:2000]

            if self.model_type == "binary":
                # Binary classification output format: [{'label': 'label_name', 'score': 0.99}]
                result = self.classifier(text)[0]
                label = result['label']
                score = result['score']
                
                # Check for human-readable labels or standard encoded ones
                valid_keywords = ['valid job requirement', 'valid', 'LABEL_0', '0'] # 0 is mapped to valid in train_bert.py
                is_valid_label = any(keyword.lower() in label.lower() for keyword in valid_keywords)
                
                is_above_threshold = score >= self.threshold
                is_valid = is_valid_label and is_above_threshold
                
                final_label = "valid" if is_valid else "junk"
                if not is_above_threshold and is_valid_label:
                    final_label = "low_confidence_valid"
                elif not is_above_threshold:
                    final_label = "low_confidence_junk"

                return {
                    'label': final_label,
                    'score': float(score),
                    'is_valid': is_valid,
                    'raw_label': label
                }
            
            else:
                # Zero-Shot Logic
                result = self.classifier(
                    text, 
                    self.candidate_labels, 
                    multi_label=False
                )
                
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                is_valid = (top_label in self.valid_labels) and (top_score >= self.threshold)
                
                final_label = "valid" if is_valid else "junk"
                if top_score < self.threshold:
                    final_label = "low_confidence_junk"

                return {
                    'label': final_label,
                    'score': float(top_score),
                    'is_valid': is_valid,
                    'raw_label': top_label
                }
            
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            return {'label': 'error', 'score': 0.0, 'is_valid': False}

    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """Classify a list of job summaries."""
        return [self.classify(t) for t in texts]
