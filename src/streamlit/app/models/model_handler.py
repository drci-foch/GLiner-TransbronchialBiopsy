import torch
from gliner import GLiNERConfig, GLiNER
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
import numpy as np
from config import config
from threading import Lock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EntityPrediction:
    """Data class for entity predictions"""
    text: str
    label: str
    score: float
    start_idx: int
    end_idx: int
    
    def to_dict(self) -> Dict:
        """Convert prediction to dictionary"""
        return {
            "text": self.text,
            "label": self.label,
            "score": round(float(self.score), 3),
            "start_idx": self.start_idx,
            "end_idx": self.end_idx
        }

class ModelHandler:
    """Handles all model-related operations"""
    
    def __init__(self):
        """Initialize the model handler"""
        self._model = None
        self._model_lock = Lock()
        self._last_used = None
        self._cache = {}
        self._cache_size = 100
        self.load_model()
    
    def load_model(self) -> None:
        """Load the GLiNER model"""
        try:
            logger.info(f"Loading model from {config.model.MODEL_PATH}")
            with self._model_lock:
                self._model = GLiNER.from_pretrained(
                    config.model.MODEL_PATH,
                    load_tokenizer=True
                )
                self._last_used = datetime.now()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload the model to free up memory"""
        with self._model_lock:
            self._model = None
            torch.cuda.empty_cache()
            logger.info("Model unloaded and CUDA cache cleared")
    
    def ensure_model_loaded(self) -> None:
        """Ensure the model is loaded before use"""
        if self._model is None:
            self.load_model()
        self._last_used = datetime.now()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before prediction"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long
        if len(text) > config.model.MAX_SEQUENCE_LENGTH:
            logger.warning(f"Text truncated from {len(text)} to {config.model.MAX_SEQUENCE_LENGTH} characters")
            text = text[:config.model.MAX_SEQUENCE_LENGTH]
        
        return text
    
    def _validate_inputs(self, text: str, labels: List[str], threshold: float) -> None:
        """Validate input parameters"""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if not labels:
            raise ValueError("Labels list cannot be empty")
        
        if not all(label in config.LABELS for label in labels):
            invalid_labels = [label for label in labels if label not in config.LABELS]
            raise ValueError(f"Invalid labels: {invalid_labels}")
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
    
    def _get_cache_key(self, text: str, labels: List[str], threshold: float) -> str:
        """Generate cache key for predictions"""
        return f"{hash(text)}_{hash(tuple(sorted(labels)))}_{threshold}"
    
    def _update_cache(self, key: str, value: List[Dict]) -> None:
        """Update prediction cache"""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, datetime.now())
    
    def predict_entities(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        threshold: float = None
    ) -> List[Dict]:
        """
        Predict entities in the given text.
        
        Args:
            text (str): Input text to analyze
            labels (List[str], optional): List of labels to detect. Defaults to all labels.
            threshold (float, optional): Confidence threshold. Defaults to config value.
        
        Returns:
            List[Dict]: List of detected entities with their labels and scores
        """
        try:
            # Set default values
            labels = labels or config.LABELS
            threshold = threshold or config.model.DEFAULT_CONFIDENCE_THRESHOLD
            
            # Validate inputs
            self._validate_inputs(text, labels, threshold)
            
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Check cache
            cache_key = self._get_cache_key(text, labels, threshold)
            if cache_key in self._cache:
                logger.info("Returning cached prediction")
                return self._cache[cache_key][0]
            
            # Ensure model is loaded
            self.ensure_model_loaded()
            
            # Make predictions
            with self._model_lock:
                raw_predictions = self._model.predict_entities(
                    text,
                    labels,
                    threshold=threshold
                )
            
            # Process predictions
            processed_predictions = []
            for pred in raw_predictions:
                entity_pred = EntityPrediction(
                    text=pred['text'],
                    label=pred['label'],
                    score=pred['score'],
                    start_idx=pred['start'],
                    end_idx=pred['end']
                )
                processed_predictions.append(entity_pred.to_dict())
            
            # Update cache
            self._update_cache(cache_key, processed_predictions)
            
            return processed_predictions
            
        except Exception as e:
            logger.error(f"Error in entity prediction: {str(e)}")
            raise
    
    def batch_predict_entities(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        threshold: float = None,
        batch_size: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Predict entities for multiple texts in batches.
        
        Args:
            texts (List[str]): List of input texts
            labels (List[str], optional): List of labels to detect
            threshold (float, optional): Confidence threshold
            batch_size (int, optional): Batch size for processing
        
        Returns:
            List[List[Dict]]: List of predictions for each input text
        """
        try:
            # Set default values
            labels = labels or config.LABELS
            threshold = threshold or config.model.DEFAULT_CONFIDENCE_THRESHOLD
            batch_size = batch_size or config.model.BATCH_SIZE
            
            all_predictions = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_predictions = [
                    self.predict_entities(text, labels, threshold)
                    for text in batch_texts
                ]
                all_predictions.extend(batch_predictions)
            
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        self.ensure_model_loaded()
        
        return {
            "model_path": config.model.MODEL_PATH,
            "last_used": self._last_used.isoformat() if self._last_used else None,
            "cache_size": len(self._cache),
            "labels": config.LABELS,
            "device": "cpu", #"cuda" if torch.cuda.is_available() else "cpu",
            "max_sequence_length": config.model.MAX_SEQUENCE_LENGTH
        }
    
    def clear_cache(self) -> None:
        """Clear the prediction cache"""
        self._cache.clear()
        logger.info("Prediction cache cleared")
    
    def __enter__(self):
        """Context manager enter"""
        self.ensure_model_loaded()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.unload_model()
