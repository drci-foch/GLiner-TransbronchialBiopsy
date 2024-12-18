import os
from pathlib import Path
import pdfplumber
from PIL import Image
import pytesseract
from typing import Optional, Tuple, Dict, List, BinaryIO, Union
import logging
from datetime import datetime
import magic  # python-magic for file type detection
import hashlib
from io import BytesIO
import shutil
import tempfile
import mimetypes
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileHandler:
    """Handles all file-related operations"""
    
    def __init__(self):
        """Initialize file handler"""
        self.config = config.file
        self.temp_dir = Path(self.config.TEMP_DIR)
        self.upload_dir = Path(self.config.UPLOAD_DIR)
        self.processed_files = {}
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Directories setup completed")
        except Exception as e:
            logger.error(f"Error setting up directories: {str(e)}")
            raise
    
    def validate_file(
        self,
        file: BinaryIO,
        filename: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file.
        
        Args:
            file: File object
            filename: Original filename
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Check file size
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
            
            if size > self.config.MAX_FILE_SIZE:
                return False, f"File size exceeds limit of {self.config.MAX_FILE_SIZE/1024/1024:.1f}MB"
            
            # Check file type
            file_type = self.get_file_type(filename)
            if not file_type:
                return False, "Unsupported file type"
            
            # Verify file content
            content = file.read()
            file.seek(0)
            
            mime = magic.Magic(mime=True)
            detected_type = mime.from_buffer(content)
            
            if file_type == 'pdf' and not detected_type.startswith('application/pdf'):
                return False, "Invalid PDF file"
            elif file_type == 'txt' and not detected_type.startswith('text/'):
                return False, "Invalid text file"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating file: {str(e)}")
            return False, str(e)
    
    def get_file_type(self, filename: str) -> Optional[str]:
        """
        Determine file type from filename.
        
        Args:
            filename: Filename to check
            
        Returns:
            Optional[str]: File type or None if unsupported
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.txt':
            return 'txt'
        return None
    
    def save_file(
        self,
        file: BinaryIO,
        filename: str,
        temp: bool = False
    ) -> Optional[Path]:
        """
        Save uploaded file.
        
        Args:
            file: File object
            filename: Original filename
            temp: Whether to save in temp directory
            
        Returns:
            Optional[Path]: Path to saved file
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            
            # Choose directory
            save_dir = self.temp_dir if temp else self.upload_dir
            file_path = save_dir / unique_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file, f)
            
            logger.info(f"File saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return None
    
    def extract_text(
        self,
        file_content: bytes,
        file_type: str
    ) -> Optional[str]:
        """
        Extract text from file content.
        
        Args:
            file_content: File content as bytes
            file_type: Type of file
            
        Returns:
            Optional[str]: Extracted text
        """
        try:
            if file_type == 'pdf':
                return self._extract_text_from_pdf(file_content)
            elif file_type == 'txt':
                return self._extract_text_from_txt(file_content)
            return None
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return None
    
    def _extract_text_from_pdf(self, content: bytes) -> Optional[str]:
        """
        Extract text from PDF content.
        
        Args:
            content: PDF content
            
        Returns:
            Optional[str]: Extracted text
        """
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            
            return text.strip() if text else None
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return None
    
    def _extract_text_from_txt(self, content: bytes) -> Optional[str]:
        """
        Extract text from TXT content.
        
        Args:
            content: TXT content
            
        Returns:
            Optional[str]: Extracted text
        """
        for encoding in config.TEXT_PROCESSING["ENCODING_ATTEMPTS"]:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with 'replace' error handler
        return content.decode('utf-8', errors='replace')
    
    def get_file_info(self, file_path: Path) -> Dict:
        """
        Get file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict: File information
        """
        try:
            stats = file_path.stat()
            
            return {
                'name': file_path.name,
                'size': stats.st_size,
                'created': datetime.fromtimestamp(stats.st_ctime),
                'modified': datetime.fromtimestamp(stats.st_mtime),
                'type': self.get_file_type(file_path.name),
                'path': str(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {}
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """
        Calculate SHA-256 hash of file content.
        
        Args:
            file_content: File content
            
        Returns:
            str: File hash
        """
        return hashlib.sha256(file_content).hexdigest()
    
    def clean_temp_files(self, max_age_hours: int = 24):
        """
        Clean old temporary files.
        
        Args:
            max_age_hours: Maximum age in hours
        """
        try:
            current_time = datetime.now()
            
            for file_path in self.temp_dir.glob('*'):
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    age_hours = (current_time - file_age).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        file_path.unlink()
                        logger.info(f"Deleted old temp file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error cleaning temp files: {str(e)}")
    
    def get_processed_files(self) -> List[Dict]:
        """
        Get list of processed files.
        
        Returns:
            List[Dict]: List of file information
        """
        try:
            processed_files = []
            
            for file_path in self.upload_dir.glob('*'):
                if file_path.is_file():
                    file_info = self.get_file_info(file_path)
                    if file_info:
                        processed_files.append(file_info)
            
            return processed_files
            
        except Exception as e:
            logger.error(f"Error getting processed files: {str(e)}")
            return []
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete file.
        
        Args:
            file_path: Path to file
            
        Returns:
            bool: True if successful
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted file: {path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.clean_temp_files()
