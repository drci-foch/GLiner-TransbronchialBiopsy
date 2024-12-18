from typing import Dict, List
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelConfig:
    """Model-related configuration settings"""
    MODEL_PATH: str = "../../finetuning/lock_models/kfold_run/fold_3/checkpoint-1000"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 32

@dataclass
class FileConfig:
    """File handling configuration settings"""
    ALLOWED_FILE_TYPES: List[str] = ("pdf", "txt")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    TEMP_DIR: str = "temp"
    UPLOAD_DIR: str = "uploads"
    CORRECTIONS_FILE: str = "corrections_log.json"
    MAX_FILES_PER_UPLOAD: int = 10

@dataclass
class UIConfig:
    """UI-related configuration settings"""
    PAGE_TITLE: str = "FochAnnot - Analyse Documents"
    PAGE_ICON: str = "🏥"
    DEFAULT_THEME: str = "light"
    MAX_DISPLAY_ROWS: int = 100
    CHART_HEIGHT: int = 400
    CHART_WIDTH: int = 800

class Config:
    """Main configuration class"""
    
    # Entity Labels
    LABELS: List[str] = [
        "Site",
        "Nombre Total De Fragments",
        "Nombre Total De Fragments Alvéolés",
        "Grade A",
        "Grade B",
        "Rejet Chronique",
        "Coloration C4d",
        "Lésion Septale",
        "Lésion Intra-Alvéolaire",
        "Éosinophilie",
        "Pneumonie Organisée",
        "DAD",
        "Infection",
        "Autre Pathologie"
    ]
    
    # Entity Colors
    COLORS: Dict[str, str] = {
        "Site": "#A1D6A3",
        "Nombre Total De Fragments": "#8EC8F2",
        "Nombre Total De Fragments Alvéolés": "#F9E26E",
        "Grade A": "#F4A3A6",
        "Grade B": "#F4C1D6",
        "Rejet Chronique": "#F6D02F",
        "Coloration C4d": "#5EC7A2",
        "Lésion Septale": "#5D9BCE",
        "Lésion Intra-Alvéolaire": "#5D8A96",
        "Éosinophilie": "#B2EBF2",
        "Pneumonie Organisée": "#D3D3D3",
        "DAD": "#F4B8D4",
        "Infection": "#FFF5BA",
        "Autre Pathologie": "#FFD2D2"
    }
    
    # Text Processing
    TEXT_PROCESSING = {
        "MAX_TEXT_LENGTH": 10000,
        "MIN_CONCLUSION_LENGTH": 50,
        "CONCLUSION_PATTERNS": [
            r"C\s*O\s*N\s*C\s*L\s*U\s*S\s*I\s*O\s*N\s*[\n\r]*",
            r"(?i)CONCLUSION[\s:]+",
            r"(?i)CONCLUSION ET SYNTHESE[\s:]+",
            r"(?i)SYNTHESE[\s:]+"
        ],
        "BIOPSY_PATTERNS": [
            r"(?:I\s*[-\s]+)?(?:B|b)iopsies?\s+(?:t|T)ransbronchiques?(?:\s*\([^)]*\))?[\s:]+",
            r"(?:I\s*[-\s]+)(?:B|b)iopsies?\s+(?:t|T)ransbronchiques?(?:\s*\([^)]*\))?",
            r"I\s*[-\s]+.*?(?:fragments?\s+biopsiques)"
        ],
        "LAVAGE_PATTERNS": [
            r"(?:II|2)\s*[-\s]+(?:L|l)avage\s+(?:b|B)roncho[\s-]*(?:a|A)lvéolaire",
            r"(?:L|l)avage\s+(?:b|B)roncho[\s-]*(?:a|A)lvéolaire"
        ],
        "ENCODING_ATTEMPTS": [
            'utf-8',
            'latin1',
            'iso-8859-1',
            'cp1252',
            'windows-1252',
            'ascii',
            'mac_roman'
        ]
    }
    
    # Export Settings
    EXPORT = {
        "DEFAULT_FILENAME": f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "CSV_ENCODING": "utf-8-sig",
        "EXCEL_SHEET_NAME": "Résultats",
        "JSON_INDENT": 2,
        "DATE_FORMAT": "%Y-%m-%d %H:%M:%S"
    }
    
    # Database Settings (if needed in future)
    DATABASE = {
        "TYPE": "sqlite",
        "NAME": "medical_annotations.db",
        "BACKUP_DIR": "backups",
        "MAX_BACKUP_COUNT": 5
    }
    
    def __init__(self):
        """Initialize configuration with sub-configs"""
        self.model = ModelConfig()
        self.file = FileConfig()
        self.ui = UIConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.file.TEMP_DIR,
            self.file.UPLOAD_DIR,
            self.DATABASE["BACKUP_DIR"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return Path(__file__).parent.parent
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path"""
        return self.project_root / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory path"""
        return self.project_root / "logs"
    
    def get_temp_file_path(self, filename: str) -> Path:
        """Get temporary file path"""
        return Path(self.file.TEMP_DIR) / filename
    
    def get_upload_file_path(self, filename: str) -> Path:
        """Get upload file path"""
        return Path(self.file.UPLOAD_DIR) / filename
    
    @staticmethod
    def load_environment_variables():
        """Load environment variables if needed"""
        required_vars = [
            # Add any required environment variables here
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
    
    def validate_configuration(self):
        """Validate configuration settings"""
        # Validate labels and colors match
        if set(self.LABELS) != set(self.COLORS.keys()):
            raise ValueError("Mismatch between defined labels and colors")
        
        # Validate directories are writable
        for directory in [self.file.TEMP_DIR, self.file.UPLOAD_DIR]:
            if not os.access(directory, os.W_OK):
                raise PermissionError(f"Directory {directory} is not writable")
        
        # Validate model path exists
        if not os.path.exists(self.model.MODEL_PATH):
            raise FileNotFoundError(f"Model path {self.model.MODEL_PATH} does not exist")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "labels": self.LABELS,
            "colors": self.COLORS,
            "model": {
                "path": self.model.MODEL_PATH,
                "threshold": self.model.DEFAULT_CONFIDENCE_THRESHOLD,
                "max_sequence_length": self.model.MAX_SEQUENCE_LENGTH,
                "batch_size": self.model.BATCH_SIZE
            },
            "file": {
                "allowed_types": self.file.ALLOWED_FILE_TYPES,
                "max_size": self.file.MAX_FILE_SIZE,
                "max_files": self.file.MAX_FILES_PER_UPLOAD
            },
            "ui": {
                "title": self.ui.PAGE_TITLE,
                "icon": self.ui.PAGE_ICON,
                "theme": self.ui.DEFAULT_THEME
            },
            "text_processing": self.TEXT_PROCESSING,
            "export": self.EXPORT,
            "database": self.DATABASE
        }

# Create global config instance
config = Config()
