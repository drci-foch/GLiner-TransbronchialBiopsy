from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Correction:
    """Data class for correction information"""
    document: str
    entity_type: str
    original_value: str
    corrected_value: str
    timestamp: str
    user: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert correction to dictionary"""
        return asdict(self)

class CorrectionsManager:
    """Manages entity corrections and correction history"""
    
    def __init__(self, corrections_file: str = None):
        """
        Initialize corrections manager.
        
        Args:
            corrections_file (str, optional): Path to corrections file
        """
        self.corrections_file = corrections_file or config.file.CORRECTIONS_FILE
        self.corrections = self.load_corrections()
    
    def load_corrections(self) -> Dict[str, List[Dict]]:
        """
        Load existing corrections from file.
        
        Returns:
            Dict[str, List[Dict]]: Loaded corrections
        """
        try:
            if Path(self.corrections_file).exists():
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
            
        except Exception as e:
            logger.error(f"Error loading corrections: {str(e)}")
            return {}
    
    def save_corrections(self) -> bool:
        """
        Save corrections to file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.corrections,
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            return True
            
        except Exception as e:
            logger.error(f"Error saving corrections: {str(e)}")
            return False
    
    def add_correction(
        self,
        document: str,
        entity_type: str,
        original_value: str,
        corrected_value: str,
        user: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Add a new correction.
        
        Args:
            document (str): Document name
            entity_type (str): Entity type
            original_value (str): Original value
            corrected_value (str): Corrected value
            user (str, optional): User making correction
            notes (str, optional): Additional notes
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            correction = Correction(
                document=document,
                entity_type=entity_type,
                original_value=original_value,
                corrected_value=corrected_value,
                timestamp=datetime.now().isoformat(),
                user=user,
                notes=notes
            )
            
            if document not in self.corrections:
                self.corrections[document] = []
            
            self.corrections[document].append(correction.to_dict())
            return self.save_corrections()
            
        except Exception as e:
            logger.error(f"Error adding correction: {str(e)}")
            return False
    
    def get_document_corrections(
        self,
        document: str
    ) -> List[Dict[str, Any]]:
        """
        Get corrections for a specific document.
        
        Args:
            document (str): Document name
            
        Returns:
            List[Dict[str, Any]]: List of corrections
        """
        return self.corrections.get(document, [])
    
    def get_entity_type_corrections(
        self,
        entity_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get corrections for a specific entity type.
        
        Args:
            entity_type (str): Entity type
            
        Returns:
            List[Dict[str, Any]]: List of corrections
        """
        corrections = []
        for doc_corrections in self.corrections.values():
            corrections.extend([
                c for c in doc_corrections 
                if c['entity_type'] == entity_type
            ])
        return corrections
    
    def get_correction_statistics(self) -> pd.DataFrame:
        """
        Generate correction statistics.
        
        Returns:
            pd.DataFrame: Correction statistics
        """
        stats = []
        
        for doc, corrections in self.corrections.items():
            doc_stats = {
                'Document': doc,
                'Total_Corrections': len(corrections)
            }
            
            # Count corrections by entity type
            entity_counts = {}
            for correction in corrections:
                entity_type = correction['entity_type']
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            doc_stats.update(entity_counts)
            stats.append(doc_stats)
        
        return pd.DataFrame(stats)
    
    def clear_document_corrections(self, document: str) -> bool:
        """
        Clear all corrections for a document.
        
        Args:
            document (str): Document name
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if document in self.corrections:
                del self.corrections[document]
                return self.save_corrections()
            return True
            
        except Exception as e:
            logger.error(f"Error clearing corrections: {str(e)}")
            return False
    
    def export_corrections(
        self,
        format: str = 'json'
    ) -> Optional[str]:
        """
        Export corrections in specified format.
        
        Args:
            format (str): Export format ('json' or 'csv')
            
        Returns:
            Optional[str]: Exported data as string
        """
        try:
            if format == 'json':
                return json.dumps(
                    self.corrections,
                    ensure_ascii=False,
                    indent=2
                )
            
            elif format == 'csv':
                rows = []
                for doc, corrections in self.corrections.items():
                    for correction in corrections:
                        row = {'Document': doc}
                        row.update(correction)
                        rows.append(row)
                
                if rows:
                    df = pd.DataFrame(rows)
                    return df.to_csv(index=False)
                
                return None
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
        except Exception as e:
            logger.error(f"Error exporting corrections: {str(e)}")
            return None
