# text_processor.py
import re
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import pdfplumber
from io import BytesIO
import pytesseract
from PIL import Image
import numpy as np
import unicodedata
from datetime import datetime
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Handles all text processing operations"""
    
    def __init__(self):
        """Initialize text processor with configuration settings"""
        self.config = config.TEXT_PROCESSING
    
    def _extract_text_from_pdf(self, content: bytes) -> Optional[str]:
        """
        Extract text from PDF content using pdfplumber.
        
        Args:
            content: PDF content
            
        Returns:
            Optional[str]: Extracted text
        """
        try:
            with pdfplumber.open(BytesIO(content)) as pdf:
                text = "\n".join(
                    page.extract_text() or "" 
                    for page in pdf.pages
                )
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
        for encoding in self.config["ENCODING_ATTEMPTS"]:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with 'replace' error handler
        return content.decode('utf-8', errors='replace')
    
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
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        try:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters while keeping French accents
            text = re.sub(
                r'[^\w\s\u00C0-\u017FàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ.,;:()[\]{}"\'-]',
                '',
                text
            )
            
            # Normalize whitespace around punctuation
            text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
            
            # Remove multiple periods
            text = re.sub(r'\.{2,}', '.', text)
            
            # Normalize dashes
            text = re.sub(r'[-‐‑‒–—―]+', '-', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return text
    
    def extract_conclusion(self, text: str) -> Optional[str]:
        """
        Extract conclusion section from text.
        
        Args:
            text (str): Full document text
            
        Returns:
            Optional[str]: Extracted conclusion section
        """
        try:
            # Clean and normalize text first
            text = self.clean_text(text)
            
            # Find conclusion section
            conclusion_text = None
            for pattern in self.config["CONCLUSION_PATTERNS"]:
                match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
                if match:
                    conclusion_text = text[match.end():]
                    logger.info("Found conclusion section")
                    break
            
            if not conclusion_text:
                logger.warning("No conclusion section found")
                return None
            
            # Extract biopsy section
            biopsy_text = None
            for pattern in self.config["BIOPSY_PATTERNS"]:
                match = re.search(pattern, conclusion_text, re.MULTILINE | re.DOTALL)
                if match:
                    start_pos = match.start()
                    section_text = conclusion_text[start_pos:]
                    
                    # Find end of biopsy section
                    end_pos = None
                    
                    # Check for lavage section
                    for lavage_pattern in self.config["LAVAGE_PATTERNS"]:
                        lavage_match = re.search(lavage_pattern, section_text)
                        if lavage_match:
                            end_pos = lavage_match.start()
                            break
                    
                    # If no lavage section, look for other end markers
                    if end_pos is None:
                        end_markers = [
                            r"(?:II|2)\s*[-\s]+",
                            r"Suresnes,",
                            r"ADICAP",
                            r"Compte-rendu",
                            r"\n\s*\n"
                        ]
                        
                        for marker in end_markers:
                            match = re.search(marker, section_text)
                            if match and match.start() > 0:
                                end_pos = match.start()
                                break
                    
                    biopsy_text = section_text[:end_pos] if end_pos else section_text
                    break
            
            if biopsy_text:
                # Clean up the extracted text
                biopsy_text = self.clean_text(biopsy_text)
                
                # Format grade notations
                biopsy_text = self._format_grade_notations(biopsy_text)
                
                return biopsy_text
            
            logger.warning("No biopsy section found in conclusion")
            return None
            
        except Exception as e:
            logger.error(f"Error in extract_conclusion: {str(e)}")
            return None
    
    def _format_grade_notations(self, text: str) -> str:
        """
        Format grade notations in text.
        
        Args:
            text (str): Text containing grade notations
            
        Returns:
            str: Text with formatted grade notations
        """
        # Handle various grade notation formats
        grade_patterns = [
            (r'A(\d|\+|x|X)B(\d|\+|x|X)', r'A\1 B\2'),  # A0B0, A1B0, AxB0, etc.
            (r'[Aa](\d|\+|x|X)[Bb](\d|\+|x|X)', r'A\1 B\2'),  # Handle lowercase
            (r'grade\s+([AaBb])(\d|\+|x|X)', r'Grade \1\2'),  # Format "grade" prefix
            (r'([AaBb])(\d|\+|x|X)\s*-\s*([AaBb])(\d|\+|x|X)', r'\1\2 \3\4')  # Handle dashes
        ]
        
        for pattern, replacement in grade_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text