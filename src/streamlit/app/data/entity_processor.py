import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Data class for entity information"""
    text: str
    label: str
    score: float
    document: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return asdict(self)

class EntityProcessor:
    """Handles processing and structuring of entity data"""
    
    def __init__(self):
        """Initialize entity processor"""
        self.labels = config.LABELS
        self.processed_entities = {}
    
    def process_entities(
        self,
        entities: List[Dict],
        filename: str,
        conclusion_text: str  # Add this parameter
    ) -> Dict[str, Any]:
        """
        Process and structure entity data.
        
        Args:
            entities (List[Dict]): List of entity predictions
            filename (str): Source document filename
            conclusion_text (str): Extracted conclusion text
            
        Returns:
            Dict[str, Any]: Structured entity data
        """
        try:
            # Initialize structured data
            structured_data = {
                'Nom_Document': filename,
                'Date_Structuration': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Conclusion': conclusion_text  # Add the conclusion text
            }
            
            # Initialize collections
            collected_entities = {label: [] for label in self.labels}
            collected_scores = {label: [] for label in self.labels}
            
            # Process each entity
            for entity in entities:
                label = entity['label']
                collected_entities[label].append(entity['text'])
                collected_scores[label].append(round(entity['score'], 3))
            
            # Create final structure
            for label in self.labels:
                structured_data[label] = (
                    ';'.join(collected_entities[label]) if collected_entities[label] else None
                )
            
            # Add scores
            if any(collected_scores.values()):
                structured_data['Scores'] = str(
                    {k: v for k, v in collected_scores.items() if v}
                )
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error processing entities: {str(e)}")
            raise
    
    def calculate_statistics(
        self,
        entities: List[Dict],
        include_scores: bool = True
    ) -> pd.DataFrame:
        """
        Calculate statistics for entities.
        
        Args:
            entities (List[Dict]): List of entity predictions
            include_scores (bool): Whether to include score statistics
            
        Returns:
            pd.DataFrame: Entity statistics
        """
        try:
            stats = []
            
            for label in self.labels:
                label_entities = [
                    entity for entity in entities 
                    if entity['label'] == label
                ]
                
                label_stats = {
                    'Label': label,
                    'Count': len(label_entities),
                    'Unique_Count': len(set(e['text'] for e in label_entities))
                }
                
                if include_scores and label_entities:
                    scores = [e['score'] for e in label_entities]
                    label_stats.update({
                        'Mean_Score': np.mean(scores),
                        'Min_Score': np.min(scores),
                        'Max_Score': np.max(scores)
                    })
                
                stats.append(label_stats)
            
            return pd.DataFrame(stats)
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise
    
    def merge_overlapping_entities(
        self,
        entities: List[Dict],
        overlap_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Merge overlapping entity predictions.
        
        Args:
            entities (List[Dict]): List of entity predictions
            overlap_threshold (float): Threshold for merging
            
        Returns:
            List[Dict]: Merged entities
        """
        try:
            if not entities:
                return []
            
            # Sort entities by start position
            sorted_entities = sorted(
                entities,
                key=lambda x: (x['start_idx'], -x['end_idx'])
            )
            
            merged = []
            current = sorted_entities[0]
            
            for next_entity in sorted_entities[1:]:
                # Calculate overlap
                overlap = (
                    min(current['end_idx'], next_entity['end_idx']) -
                    max(current['start_idx'], next_entity['start_idx'])
                )
                
                union = (
                    max(current['end_idx'], next_entity['end_idx']) -
                    min(current['start_idx'], next_entity['start_idx'])
                )
                
                overlap_ratio = overlap / union if union > 0 else 0
                
                if overlap_ratio > overlap_threshold:
                    # Merge entities
                    if current['score'] < next_entity['score']:
                        current = next_entity
                else:
                    merged.append(current)
                    current = next_entity
            
            merged.append(current)
            return merged
            
        except Exception as e:
            logger.error(f"Error merging entities: {str(e)}")
            raise
    
    def validate_entities(self, entities: List[Dict]) -> List[str]:
        """
        Validate entity predictions.
        
        Args:
            entities (List[Dict]): List of entity predictions
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        try:
            for entity in entities:
                # Check required fields
                required_fields = ['text', 'label', 'score']
                missing_fields = [
                    field for field in required_fields 
                    if field not in entity
                ]
                if missing_fields:
                    errors.append(
                        f"Missing required fields: {', '.join(missing_fields)}"
                    )
                
                # Validate label
                if 'label' in entity and entity['label'] not in self.labels:
                    errors.append(f"Invalid label: {entity['label']}")
                
                # Validate score
                if 'score' in entity:
                    score = entity['score']
                    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                        errors.append(f"Invalid score: {score}")
            
            return errors
            
        except Exception as e:
            logger.error(f"Error validating entities: {str(e)}")
            return [str(e)]

