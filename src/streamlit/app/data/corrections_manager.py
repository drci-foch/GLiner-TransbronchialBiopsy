import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import pandas as pd
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CorrectionsManager:
    def __init__(self, base_log_dir: str = "correction_logs"):
        """Initialize corrections manager"""
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(exist_ok=True)
        self.corrections = {}
        
        # Will be set when user logs in
        self.current_user = None
        self.session_id = None
        self.corrections_file = None

    def set_user(self, username: str):
        """Set current user and initialize their log file"""
        self.current_user = username
        
        # Create user directory
        user_dir = self.base_log_dir / username
        user_dir.mkdir(exist_ok=True)
        
        # Create new session log file
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.corrections_file = user_dir / f"corrections_log_{self.session_id}.json"
        
        # Load existing corrections
        self.corrections = {}
    
    def get_user_logs(self, username: str) -> List[Path]:
        """Get all log files for a user"""
        user_dir = self.base_log_dir / username
        if user_dir.exists():
            return sorted(user_dir.glob("*.json"), reverse=True)
        return []
    
    def add_correction(
        self,
        document: str,
        entity_type: str,
        original_value: str,
        corrected_value: str,
        full_row_data: pd.Series,
        user: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        """
        Add a new correction and update latest state.
        """
        try:
            # Initialize document entry if it doesn't exist
            if document not in self.corrections:
                self.corrections[document] = {
                    "history": [],
                    "latest_state": None
                }
            
            # Create correction entry
            correction = {
                "document": document,
                "entity_type": entity_type,
                "original_value": original_value,
                "corrected_value": corrected_value,
                "timestamp": datetime.now().isoformat(),
                "user": user,
                "notes": notes
            }
            
            # Add to history
            self.corrections[document]["history"].append(correction)
            
            # Update latest state
            if self.corrections[document]["latest_state"] is None:
                # If first correction, use full row data as base
                latest_state = full_row_data.to_dict()
            else:
                # Use existing latest state
                latest_state = self.corrections[document]["latest_state"].copy()
            
            # Apply the new correction
            latest_state[entity_type] = corrected_value
            latest_state["last_updated"] = correction["timestamp"]
            
            # Store the updated state
            self.corrections[document]["latest_state"] = latest_state
            
            return self.save_corrections()
            
        except Exception as e:
            logger.error(f"Error adding correction: {str(e)}")
            return False
    
    def save_corrections(self) -> bool:
        """Save corrections to current session log file"""
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(self.corrections, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving corrections: {str(e)}")
            return False
    
    def clear_session(self):
        """Clear current session corrections"""
        self.corrections = {}
        # Create new session log file
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.corrections_file = self.base_log_dir / f"corrections_log_{self.session_id}.json"
    
    def get_latest_state(self, document: str) -> Optional[Dict]:
        """Get the latest state of a document with all corrections applied."""
        if document in self.corrections:
            return self.corrections[document]["latest_state"]
        return None
    
    def get_correction_history(self, document: str) -> List[Dict]:
        """Get the correction history for a document."""
        if document in self.corrections:
            return self.corrections[document]["history"]
        return []
    
    def export_corrections(self, format: str = 'json') -> Optional[str]:
        """Export corrections with latest states."""
        try:
            if format == 'json':
                # Create export structure focusing on latest states
                export_data = {
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "corrections": self.corrections
                }
                return json.dumps(export_data, ensure_ascii=False, indent=2)
                
            elif format == 'csv':
                # Create DataFrame from latest states
                rows = []
                for doc, data in self.corrections.items():
                    if data["latest_state"]:
                        row = data["latest_state"].copy()
                        row["document"] = doc
                        row["total_corrections"] = len(data["history"])
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