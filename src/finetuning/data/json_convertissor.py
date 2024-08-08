import pandas as pd
import json

excel_file = './BTB_annotations_update.xlsx'
df = pd.read_excel(excel_file)

json_records = []

for _, row in df.iterrows():
    record = {
        "tokenized_text": row['Free-text (conclusion)'].split(),
        "ner": {}
    }
    
    columns = [
        'Site', 'Total number of fragments', 'Number of alveolated fragments',
        'A grade', 'B grade', 'Chronic rejection', 'C4d staining',
        'Septal injury', 'Intra-alveolar injury', 'Eosinophilia',
        'Organising pneumonia', 'DAD', 'Infection', 'Other pathology'
    ]
    
    for col in columns:
        value = row[col]
        if pd.notna(value):
            record["ner"][col] = value

    json_records.append(record)

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(json_records, f, ensure_ascii=False, indent=4)
print("Data has been converted to JSON format and saved to 'output.json'")

