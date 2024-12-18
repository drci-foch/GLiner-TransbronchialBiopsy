import re
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path
import pdfplumber
from io import BytesIO
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from config import config
import unicodedata
import fitz  # PyMuPDF
import traceback

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
        self.last_processed = {}  # Cache for processed texts
    
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
            text = re.sub(r'[^\w\s\u00C0-\u017FàâäéèêëîïôöùûüÿçÀÂÄÉÈÊËÎÏÔÖÙÛÜŸÇ.,;:()[\]{}"\'-]', '', text)
            
            # Normalize whitespace around punctuation
            text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
            
            # Remove multiple periods
            text = re.sub(r'\.{2,}', '.', text)
            
            # Normalize dashes
            text = re.sub(r'[-‐‑‒–—―]+', '-', text)
            
            # Fix common OCR errors
            text = self._fix_ocr_errors(text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in clean_text: {str(e)}")
            return text
    
    def _fix_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in text.
        
        Args:
            text (str): Text with potential OCR errors
            
        Returns:
            str: Text with corrected OCR errors
        """
        # Common OCR error corrections
        corrections = {
            r'([A-Za-z])0': r'\1O',  # Replace 0 with O when after a letter
            r'1l': 'Il',  # Replace 1l with Il
            r'\bI([^a-zA-Z])': r'1\1',  # Replace I with 1 when followed by non-letter
            r'rn': 'm',  # Replace 'rn' with 'm'
            r'([0-9])l': r'\1I',  # Replace l with I when after a number
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def extract_conclusion(self, text: str) -> Optional[str]:
        """
        Extract conclusion section from text.
        
        Args:
            text (str): Full document text
            
        Returns:
            Optional[str]: Extracted conclusion section or None if not found
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
    
    def extract_text_from_file(self, file_content: bytes, file_type: str) -> Optional[str]:
        """
        Extract text from uploaded file.
        
        Args:
            file_content (bytes): File content
            file_type (str): File type (pdf or txt)
            
        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            if file_type == "pdf":
                return self._extract_text_from_pdf(file_content)
            elif file_type == "txt":
                return self._extract_text_from_txt(file_content)
            else:
                logger.error(f"Unsupported file type: {file_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_type} file: {str(e)}")
            return None
    
    def _extract_text_from_pdf(self, file_content: bytes) -> Optional[str]:
        """
        Extract text from PDF file.
        
        Args:
            file_content (bytes): PDF file content
            
        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            # Try pdfplumber first
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages)
                
                if text.strip():
                    return text
            
            # If pdfplumber fails or returns empty text, try PyMuPDF
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if text.strip():
                return text
            
            # If both fail, try OCR
            return self._extract_text_with_ocr(file_content)
            
        except Exception as e:
            logger.error(f"Error in PDF text extraction: {str(e)}")
            return None
    
    def _extract_text_from_txt(self, file_content: bytes) -> Optional[str]:
        """
        Extract text from TXT file.
        
        Args:
            file_content (bytes): TXT file content
            
        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        for encoding in self.config["ENCODING_ATTEMPTS"]:
            try:
                return file_content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with 'replace' error handler
        return file_content.decode('utf-8', errors='replace')
    
    def _extract_text_with_ocr(self, file_content: bytes) -> Optional[str]:
        """
        Extract text using OCR when other methods fail.
        
        Args:
            file_content (bytes): File content
            
        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        try:
            # Convert PDF to images
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Perform OCR
                page_text = pytesseract.image_to_string(
                    img,
                    lang='fra',
                    config='--psm 1 --oem 3'
                )
                text += page_text + "\n"
            
            doc.close()
            return text if text.strip() else None
            
        except Exception as e:
            logger.error(f"Error in OCR text extraction: {str(e)}")
            return None
    
    def validate_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate extracted text.
        
        Args:
            text (str): Text to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not text:
            return False, "Text is empty"
        
        if len(text) < self.config["MIN_CONCLUSION_LENGTH"]:
            return False, f"Text is too short (minimum {self.config['MIN_CONCLUSION_LENGTH']} characters)"
        
        if len(text) > self.config["MAX_TEXT_LENGTH"]:
            return False, f"Text is too long (maximum {self.config['MAX_TEXT_LENGTH']} characters)"
        
        return True, None
    
    def get_text_statistics(self, text: str) -> Dict:
        """
        Get statistics about the text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict: Text statistics
        """
        if not text:
            return {}
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "average_word_length": np.mean([len(word) for word in words]),
            "average_sentence_length": np.mean([len(sent.split()) for sent in sentences if sent.strip()])
        }
