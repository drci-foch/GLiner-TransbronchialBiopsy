import pandas as pd
import json
import nltk
from typing import List, Tuple, Dict
import re

def tokenize_text(text: str) -> List[str]:
    """Tokenize French medical text while preserving punctuation."""
    # Simple tokenization by whitespace and punctuation
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens

def find_token_spans(text: str, entity: str) -> List[Tuple[int, int, str]]:
    """Find token spans for an entity in the text."""
    if pd.isna(entity) or not entity or entity == []:
        return []
    
    text_tokens = tokenize_text(text)
    entity_tokens = tokenize_text(str(entity))
    spans = []
    
    # Convert tokens to lowercase for comparison
    text_tokens_lower = [t.lower() for t in text_tokens]
    entity_tokens_lower = [t.lower() for t in entity_tokens]
    
    for i in range(len(text_tokens)):
        matches = True
        for j in range(len(entity_tokens)):
            if (i + j >= len(text_tokens) or 
                text_tokens_lower[i + j] != entity_tokens_lower[j]):
                matches = False
                break
        if matches:
            spans.append((i, i + len(entity_tokens) - 1, entity))
    
    return spans

def convert_to_json(excel_file: str) -> List[Dict]:
    # Load Excel file
    df = pd.read_excel(excel_file)
    
    # Column mappings (French labels)
    column_translation = {
        'Site': 'site',
        'Total number of fragments': 'nombre_total_de_fragments',
        'Number of alveolated fragments': 'nombre_total_de_fragments_alveoles',
        'A grade': 'grade_a',
        'B grade': 'grade_b',
        'Chronic rejection': 'rejet_chronique',
        'C4d staining': 'coloration_c4d',
        'Septal injury': 'lesion_septale',
        'Intra-alveolar injury': 'lesion_intra_alveolaire',
        'Eosinophilia': 'eosinophilie',
        'Organising pneumonia': 'pneumonie_organisee',
        'DAD': 'dad',
        'Infection': 'infection',
        'Other pathology': 'autre_pathologie'
    }

    json_records = []

    for idx, row in df.iterrows():
        text = row['Free-text (conclusion)']
        if pd.isna(text):
            continue
            
        tokens = tokenize_text(text)
        ner_spans = []
        
        # Process each column and find spans
        for col, french_col in column_translation.items():
            value = row[col]
            if pd.notna(value) and value != []:
                # Handle multiple values separated by semicolon
                if isinstance(value, str) and ';' in value:
                    values = [v.strip() for v in value.split(';')]
                else:
                    values = [value]
                
                for val in values:
                    spans = find_token_spans(text, str(val))
                    for start, end, _ in spans:
                        ner_spans.append([start, end, french_col])
        
        record = {
            "tokenized_text": tokens,
            "ner": ner_spans
        }
        json_records.append(record)
        
        # Print progress
        print(f"Processed record {idx + 1}/{len(df)}")

    return json_records

if __name__ == "__main__":
    excel_file = './BTB_annotations.xlsx'
    json_records = convert_to_json(excel_file)
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(json_records, f, ensure_ascii=False, indent=4)
    
    print("Data has been converted to JSON format and saved to 'data.json'")