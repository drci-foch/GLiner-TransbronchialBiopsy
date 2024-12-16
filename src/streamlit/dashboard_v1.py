import streamlit as st
import os
import torch
from gliner import GLiNERConfig, GLiNER
import re
import pandas as pd
import plotly.graph_objects as go
import pdfplumber
from io import BytesIO
import base64
from typing import Dict, List
import tempfile
import mimetypes
import json
from datetime import datetime

class MedicalAnnotationDashboard:
    def __init__(self):
        self.model = GLiNER.from_pretrained("../finetuning/models/kfold_run/fold_3/checkpoint-1000", load_tokenizer=True)
        
        self.labels = [
            "Site", "Nombre Total De Fragments", "Nombre Total De Fragments Alvéolés",
            "Grade A", "Grade B", "Rejet Chronique", "Coloration C4d", "Lésion Septale",
            "Lésion Intra-Alvéolaire", "Éosinophilie", "Pneumonie Organisée", "DAD",
            "Infection", "Autre Pathologie"
        ]
        self.colors = {
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
        # Initialize session state
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = {}
        if 'results_df' not in st.session_state:
            st.session_state.results_df = None
        if 'corrections' not in st.session_state:
            st.session_state.corrections = {}
        
        # Initialize corrections file path
        self.corrections_file = "corrections_log.json"
        self.load_corrections()

    def highlight_entities(self, text: str, entities: List[Dict]) -> str:
        """Highlight entities in text with custom styling"""
        for entity in entities:
            label = entity["label"]
            entity_text = entity["text"]
            color = self.colors.get(label, "#F5F5F5")
            
            replacement = f'''<span class="highlighted-entity" 
                                  style="background-color: {color};">
                                {entity_text}
                                <span class="entity-tooltip">{label}</span>
                            </span>'''
            
            text = re.sub(
                rf"({re.escape(entity_text)})",
                replacement,
                text
            )
        
        return text

    def create_entity_stats(self, entities: List[Dict]) -> pd.DataFrame:
        """Create statistics from identified entities"""
        entity_counts = {}
        for entity in entities:
            label = entity["label"]
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        stats_df = pd.DataFrame(
            [(label, entity_counts.get(label, 0)) for label in self.labels],
            columns=['Entity', 'Count']
        )
        return stats_df

    def create_entity_distribution_chart(self, stats_df: pd.DataFrame):
        """Create an interactive bar chart of entity distribution"""
        fig = go.Figure(data=[
            go.Bar(
                x=stats_df['Count'],
                y=stats_df['Entity'],
                orientation='h',
                marker_color=[self.colors[entity] for entity in stats_df['Entity']],
                hovertemplate="<b>%{y}</b><br>" +
                             "Count: %{x}<br>" +
                             "<extra></extra>"
            )
        ])
        
        fig.update_layout(
            title="Distribution des Entités Identifiées",
            xaxis_title="Nombre d'occurrences",
            yaxis_title="Type d'entité",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def extract_text_from_file(self, file_content, file_type):
        """Extract text from uploaded file based on file type"""
        try:
            if file_type == "pdf":
                with pdfplumber.open(BytesIO(file_content)) as pdf:
                    return "\n".join(page.extract_text() for page in pdf.pages)
            elif file_type == "txt":
                # Try different encodings
                encodings = [
                    'utf-8', 'latin1', 'iso-8859-1', 'cp1252', 
                    'windows-1252', 'ascii', 'mac_roman'
                ]
                
                for encoding in encodings:
                    try:
                        return file_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue
                        
                # If all encodings fail, try with errors='replace'
                print(file_content.decode('utf-8', errors='replace'))
                return file_content.decode('utf-8', errors='replace')
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            st.error(f"Error extracting text from {file_type.upper()} file: {str(e)}")
            return None

    def get_file_type(self, filename):
        """Determine file type from filename"""
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext == '.txt':
            return 'txt'
        return None

    def display_file_content(self, file_content, filename):
        """Display file content based on file type"""
        file_type = self.get_file_type(filename)
        
        with st.expander(f"📄 Visualiser {filename}"):
            if file_type == 'pdf':
                base64_pdf = base64.b64encode(file_content).decode('utf-8')
                pdf_display = f'''
                    <iframe src="data:application/pdf;base64,{base64_pdf}" 
                            width="100%" 
                            height="800px" 
                            type="application/pdf">
                    </iframe>
                '''
                st.markdown(pdf_display, unsafe_allow_html=True)
            elif file_type == 'txt':
                st.text_area("Contenu du fichier", file_content.decode('utf-8'), 
                            height=400, disabled=True)

    def extract_conclusion(self, text):
        """Extract only the biopsy section from the conclusion with debug prints"""
        # First find the CONCLUSION section
        conclusion_patterns = [
            r"C\s*O\s*N\s*C\s*L\s*U\s*S\s*I\s*O\s*N\s*[\n\r]*",
            r"(?i)CONCLUSION[\s:]+",
            r"(?i)CONCLUSION ET SYNTHESE[\s:]+",
            r"(?i)SYNTHESE[\s:]+"
        ]
        
        # Print the original text
        print("\n=== Original Text ===")
        print(text)
        
        # Find conclusion section
        conclusion_text = None
        for pattern in conclusion_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                conclusion_text = text[match.end():]
                print("\n=== Found Conclusion Section ===")
                print(conclusion_text)
                break
        
        if not conclusion_text:
            print("\nNo conclusion section found!")
            return None

        # Pattern for various forms of biopsy section, including Roman numerals
        biopsy_start_patterns = [
            r"(?:I\s*[-\s]+)?(?:B|b)iopsies?\s+(?:t|T)ransbronchiques?(?:\s*\([^)]*\))?[\s:]+",
            r"(?:I\s*[-\s]+)(?:B|b)iopsies?\s+(?:t|T)ransbronchiques?(?:\s*\([^)]*\))?",
            r"I\s*[-\s]+.*?(?:fragments?\s+biopsiques)"  # New pattern for this case
        ]
        
        # Pattern for lavage section
        lavage_patterns = [
            r"(?:II|2)\s*[-\s]+(?:L|l)avage\s+(?:b|B)roncho[\s-]*(?:a|A)lvéolaire",
            r"(?:L|l)avage\s+(?:b|B)roncho[\s-]*(?:a|A)lvéolaire"
        ]

        # Find biopsy section
        biopsy_text = None
        for bstart_pattern in biopsy_start_patterns:
            match = re.search(bstart_pattern, conclusion_text, re.MULTILINE | re.DOTALL)
            if match:
                start_pos = match.start()
                section_text = conclusion_text[start_pos:]
                print("\n=== Found Biopsy Section Start ===")
                print(section_text)
                
                # Look for the next section (lavage or end)
                end_pos = None
                
                # Check for lavage section
                for lavage_pattern in lavage_patterns:
                    lavage_match = re.search(lavage_pattern, section_text)
                    if lavage_match:
                        end_pos = lavage_match.start()
                        print("\n=== Found Lavage Section (End Marker) ===")
                        print(f"Ends at position: {end_pos}")
                        break
                
                # If no lavage section found, look for other end markers
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
                            print("\n=== Found End Marker ===")
                            print(f"Marker: {marker}, Position: {end_pos}")
                            break
                
                # Extract the text
                biopsy_text = section_text[:end_pos] if end_pos else section_text
                break
        
        if biopsy_text:
            # Clean up the extracted text
            biopsy_text = re.sub(r'\s+', ' ', biopsy_text)
            biopsy_text = biopsy_text.strip('.- \t\n\r')
            
            # Format grade notations - add space between grades
            # Handle various grade notation formats
            grade_patterns = [
                (r'A(\d|\+|x|X)B(\d|\+|x|X)', r'A\1 B\2'),  # A0B0, A1B0, AxB0, etc.
                (r'[Aa](\d|\+|x|X)[Bb](\d|\+|x|X)', r'A\1 B\2')  # Handle lowercase variations
            ]
            
            for pattern, replacement in grade_patterns:
                biopsy_text = re.sub(pattern, replacement, biopsy_text)
            
            # Remove any extra spaces that might have been created
            biopsy_text = re.sub(r'\s+', ' ', biopsy_text).strip()
            
            print("\n=== Final Extracted Text ===")
            print(biopsy_text)
            return biopsy_text
        
        print("\nNo biopsy section found!")
        return None
    
    def create_structured_data(self, entities, filename):
        """Convert entities to structured format with filename, handling multiple entities per label"""
        entity_dict = {
            'Nom_Document': filename,
            'Date_Structuration': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Initialize dictionary to collect multiple entities per label
        collected_entities = {}
        collected_scores = {}
        
        # Initialize all labels - Notice we're not adding _Score suffix to all columns
        for label in self.labels:
            entity_dict[label] = None
            collected_entities[label] = []
            collected_scores[label] = []
        
        # Add a single Scores column
        entity_dict['Scores'] = {}
        
        # Collect all entities and scores by label
        for entity in entities:
            label = entity['label']
            collected_entities[label].append(entity['text'])
            collected_scores[label].append(round(entity['score'], 3))
        
        # Join multiple entities with semicolons and store scores in a more compact way
        for label in self.labels:
            if collected_entities[label]:
                entity_dict[label] = ";".join(collected_entities[label])
                # Store scores in a more compact format
                if collected_scores[label]:
                    entity_dict['Scores'][label] = [round(score, 2) for score in collected_scores[label]]
        
        # Convert scores dictionary to string representation
        entity_dict['Scores'] = str(entity_dict['Scores'])
        
        return entity_dict

    def update_results_dataframe(self, new_data):
        """Update the results dataframe with new data"""
        new_row = pd.DataFrame([new_data])
        if st.session_state.results_df is None:
            st.session_state.results_df = new_row
        else:
            st.session_state.results_df = pd.concat([st.session_state.results_df, new_row], ignore_index=True)

    def get_table_download_link(self, df):
        """Generate download link for dataframe"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        filename = f"resultats_analyse_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">📥 Télécharger les résultats (CSV)</a>'
        return href

    def setup_page_config(self):
        """Configure the Streamlit page settings"""
        st.set_page_config(
            page_title="FochAnnot - Analyse Documents",
            page_icon="🏥",
            layout="wide"
        )

    def apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
            <style>
                .main {
                    padding: 2rem;
                }
                .file-uploader {
                    margin: 2rem 0;
                    padding: 2rem;
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    text-align: center;
                }
                .results-container {
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                }
                .download-button {
                    display: inline-block;
                    margin: 1rem 0;
                    padding: 0.75rem 1.5rem;
                    background-color: #4CAF50;
                    color: white !important;
                    border-radius: 4px;
                    text-decoration: none;
                    transition: background-color 0.3s ease;
                }
                .download-button:hover {
                    background-color: #45a049;
                }
                .highlighted-entity {
                    cursor: help;
                    border-radius: 4px;
                    padding: 2px 4px;
                    display: inline-block;
                    transition: all 0.3s ease;
                    position: relative;
                }
                .highlighted-entity:hover {
                    transform: scale(1.05);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .entity-tooltip {
                    visibility: hidden;
                    background-color: #333;
                    color: white;
                    text-align: center;
                    padding: 5px 10px;
                    border-radius: 6px;
                    position: absolute;
                    z-index: 1000;
                    bottom: 125%;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 12px;
                    white-space: nowrap;
                }
                .highlighted-entity:hover .entity-tooltip {
                    visibility: visible;
                }
                .stDataFrame {
                    margin: 1rem 0;
                }
                .document-viewer {
                    margin-top: 1rem;
                    padding: 1rem;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                .viewer-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                }
                .close-button {
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    background-color: #ff4b4b;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                .close-button:hover {
                    background-color: #ff3333;
                }
                iframe {
                    border: none;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
            </style>
        """, unsafe_allow_html=True)

    def display_file_modal(self, file_content, filename):
        """Display file content in a dialog"""
        if st.button(f"📄 Voir {filename}", key=f"btn_{filename}"):
            # Create a dialog using columns
            with st.expander(f"📄 {filename}", expanded=True):
                # Close button
                col1, col2 = st.columns([0.9, 0.1])
                with col2:
                    if st.button("❌", key=f"close_{filename}"):
                        st.rerun()
                
                file_type = self.get_file_type(filename)
                
                if file_type == 'pdf':
                    base64_pdf = base64.b64encode(file_content).decode('utf-8')
                    pdf_display = f'''
                        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                                width="100%" 
                                height="800px" 
                                type="application/pdf">
                        </iframe>
                    '''
                    st.markdown(pdf_display, unsafe_allow_html=True)
                elif file_type == 'txt':
                    # Try different encodings
                    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 
                            'windows-1252', 'ascii', 'mac_roman']
                    text_content = None
                    
                    for encoding in encodings:
                        try:
                            text_content = file_content.decode(encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if text_content is None:
                        text_content = file_content.decode('utf-8', errors='replace')
                    
                    st.text_area("", text_content, height=800)

    def get_excel_download_link(self, df):
        """Generate download link for Excel file"""
        # Create a BytesIO object to store the Excel file
        output = BytesIO()
        
        # Create Excel writer object with xlsxwriter engine
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Résultats')
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Résultats']
            
            # Add some styling
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D9EAD3',
                'border': 1
            })
            
            # Write the column headers with the header format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Auto-adjust columns' width
            for idx, col in enumerate(df.columns):
                series = df[col]
                max_len = max(
                    series.astype(str).apply(len).max(),
                    len(str(series.name))
                ) + 2
                worksheet.set_column(idx, idx, max_len)

        # Generate download link
        b64 = base64.b64encode(output.getvalue()).decode()
        filename = f"resultats_analyse_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button">📥 Télécharger les résultats (Excel)</a>'
        return href


    def load_corrections(self):
        """Load existing corrections from file"""
        try:
            if os.path.exists(self.corrections_file):
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    st.session_state.corrections = json.load(f)
        except Exception as e:
            st.error(f"Error loading corrections: {str(e)}")

    def save_corrections(self):
        """Save corrections to file"""
        try:
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump(st.session_state.corrections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"Error saving corrections: {str(e)}")

    def add_correction(self, document_name: str, entity_type: str, original_value: str, corrected_value: str):
        """Add a correction to the corrections log"""
        correction = {
            "timestamp": datetime.now().isoformat(),
            "document": document_name,
            "entity_type": entity_type,
            "original_value": original_value,
            "corrected_value": corrected_value
        }
        
        if document_name not in st.session_state.corrections:
            st.session_state.corrections[document_name] = []
        
        st.session_state.corrections[document_name].append(correction)
        self.save_corrections()

    def display_correction_interface(self, row_index: int, display_df: pd.DataFrame):
        """Display correction interface for a specific row"""
        document_name = display_df.iloc[row_index]['Nom_Document']
        
        st.markdown(f"### Corrections pour {document_name}")
        
        # Create columns for better layout
        cols = st.columns(2)
        
        with cols[0]:
            # Entity selection
            entity_type = st.selectbox(
                "Sélectionner l'entité à corriger",
                self.labels,
                key=f"entity_select_{row_index}"
            )
            
            current_value = display_df.iloc[row_index][entity_type]
            current_value = str(current_value) if pd.notna(current_value) else ""
            
            # Display current value
            st.text_area(
                "Valeur actuelle",
                current_value,
                disabled=True,
                key=f"current_value_{row_index}"
            )
            
            # Input for correction
            corrected_value = st.text_area(
                "Valeur corrigée",
                key=f"correction_input_{row_index}"
            )
            
            # Submit correction
            if st.button("Soumettre la correction", key=f"submit_{row_index}"):
                if corrected_value != current_value:
                    self.add_correction(
                        document_name,
                        entity_type,
                        current_value,
                        corrected_value
                    )
                    st.success("Correction enregistrée!")
                    
                    # Update the display dataframe
                    display_df.at[row_index, entity_type] = corrected_value
                    st.session_state.results_df = display_df
        
        with cols[1]:
            # Display correction history for this document
            st.markdown("### Historique des corrections")
            if document_name in st.session_state.corrections:
                for correction in st.session_state.corrections[document_name]:
                    with st.expander(
                        f"{correction['entity_type']} - {correction['timestamp'][:16]}"
                    ):
                        st.markdown(f"**Original:** {correction['original_value']}")
                        st.markdown(f"**Corrigé:** {correction['corrected_value']}")

    def get_corrections_download_link(self):
        """Generate download link for corrections log"""
        if st.session_state.corrections:
            json_str = json.dumps(st.session_state.corrections, ensure_ascii=False, indent=2)
            b64 = base64.b64encode(json_str.encode('utf-8')).decode()
            filename = f"corrections_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="download-button">📥 Télécharger l\'historique des corrections (JSON)</a>'
            return href
        return None
        
    def run(self):
        """Main application logic"""
        self.setup_page_config()
        self.apply_custom_css()

        st.title("🏥 FochAnnot : Analyse de Documents")

        with st.expander("📖 Instructions d'utilisation", expanded=False):
            st.markdown("""
                ### Comment utiliser FochAnnot :
                1. **Upload de fichiers** : Téléchargez vos documents (PDF ou TXT)
                2. **Extraction** : Le système extraira automatiquement la conclusion
                3. **Analyse** : Les entités seront détectées et structurées
                4. **Corrections** : Identifiez et corrigez les erreurs du modèle
                5. **Export** : Téléchargez les résultats et l'historique des corrections
                
                ### Fonctionnalités :
                - Support des fichiers PDF et TXT
                - Détection automatique des conclusions
                - Interface de correction des erreurs
                - Visualisation interactive des documents
                - Export des résultats en Excel
                - Export de l'historique des corrections
            """)

        # Sidebar controls
        with st.sidebar:
            st.header("Configuration")
            threshold = st.slider(
                "Seuil de confiance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Ajustez le seuil de confiance pour la détection des entités"
            )
            
            if st.session_state.results_df is not None:
                st.markdown("### Actions")
                if st.button("🗑️ Effacer tous les résultats"):
                    st.session_state.results_df = None
                    st.session_state.processed_files = {}
                    st.rerun()

        # File upload
        uploaded_files = st.file_uploader(
            "Choisissez un ou plusieurs fichiers (PDF ou TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Glissez-déposez vos fichiers ou cliquez pour les sélectionner"
        )

        if uploaded_files:
            with st.spinner("Traitement des documents en cours..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        # Process new file
                        file_content = uploaded_file.read()
                        file_type = self.get_file_type(uploaded_file.name)
                        
                        if file_type:
                            st.session_state.processed_files[uploaded_file.name] = file_content
                            
                            # Extract and process text
                            file_text = self.extract_text_from_file(file_content, file_type)
                            if file_text:
                                conclusion = self.extract_conclusion(file_text)
                                if conclusion:
                                    entities = self.model.predict_entities(
                                        conclusion,
                                        self.labels,
                                        threshold=threshold
                                    )
                                    if entities:
                                        # Create and update structured data
                                        structured_data = self.create_structured_data(
                                            entities,
                                            uploaded_file.name
                                        )
                                        self.update_results_dataframe(structured_data)
                                        
                                        # Show success message
                                        st.success(f"✅ {uploaded_file.name} traité avec succès")
                                    else:
                                        st.warning(f"⚠️ Aucune entité identifiée dans {uploaded_file.name}")
                                else:
                                    st.error(f"❌ Pas de conclusion trouvée dans {uploaded_file.name}")
                            else:
                                st.error(f"❌ Impossible de lire {uploaded_file.name}")
                        else:
                            st.error(f"❌ Type de fichier non supporté pour {uploaded_file.name}")

        # Display results if available
        if st.session_state.results_df is not None:
            st.markdown("---")
            st.subheader("Résultats d'analyse")
            
            # Display tabs for different views
            tab1, tab2, tab3 = st.tabs([
                "📊 Données structurées",
                "📈 Statistiques",
                "✏️ Corrections"
            ])
            
            with tab1:
                # Configure column display order and format
                column_order = ['Nom_Document', 'Date_Structuration']
                for label in self.labels:
                    column_order.append(label)
                column_order.append('Scores')
                
                # Reorder columns
                display_df = st.session_state.results_df[column_order]
                
                # Display interactive dataframe
                st.dataframe(
                    display_df,
                    column_config={
                        "Nom_Document": st.column_config.Column(
                            "Document",
                            help="Cliquez pour voir le document",
                            width="medium",
                        ),
                        "Date_Structuration": st.column_config.DatetimeColumn(
                            "Date d'analyse",
                            help="Date et heure de l'analyse",
                            format="DD/MM/YYYY HH:mm",
                            width="medium",
                        ),
                        "Scores": st.column_config.Column(
                            "Scores de confiance",
                            help="Scores de confiance pour chaque entité détectée",
                            width="medium",
                            disabled=True
                        )
                    },
                    hide_index=True,
                )
                
                # Add modal viewers for each file
                for idx, row in display_df.iterrows():
                    filename = row['Nom_Document']
                    if filename in st.session_state.processed_files:
                        file_content = st.session_state.processed_files[filename]
                        self.display_file_modal(file_content, filename)

                # Download buttons with proper styling
                st.markdown("### Export des résultats")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(self.get_excel_download_link(display_df), unsafe_allow_html=True)
            
            with tab2:
                if len(display_df) > 0:
                    # Create statistics and visualizations
                    col_stats1, col_stats2 = st.columns([1, 1])
                    
                    with col_stats1:
                        st.markdown("### Distribution des entités par document")
                        for filename in display_df['Nom_Document'].unique():
                            st.markdown(f"**Document : {filename}**")
                            doc_data = display_df[display_df['Nom_Document'] == filename]
                            entities_found = [col for col in self.labels if doc_data[col].notna().any()]
                            if entities_found:
                                for entity in entities_found:
                                    value = doc_data[entity].iloc[0]
                                    if pd.notna(value):  # Only show non-null values
                                        st.markdown(f"- {entity}: {value}")
                            else:
                                st.markdown("*Aucune entité détectée*")
                            st.markdown("---")
                    
                    with col_stats2:
                        st.markdown("### Statistiques globales")
                        total_docs = len(display_df)
                        total_entities = sum(display_df[label].notna().sum() for label in self.labels)
                        
                        st.metric("Documents analysés", total_docs)
                        st.metric("Entités détectées", total_entities)
                        if total_docs > 0:
                            st.metric(
                                "Moyenne d'entités par document",
                                f"{total_entities/total_docs:.1f}"
                            )

                        # Add correction statistics
                        if st.session_state.corrections:
                            total_corrections = sum(
                                len(corrs) for corrs in st.session_state.corrections.values()
                            )
                            st.metric("Corrections effectuées", total_corrections)
                else:
                    st.info("Aucune donnée disponible pour les statistiques")
            
            with tab3:
                st.markdown("### Interface de correction")
                st.markdown("""
                    Cette interface permet de corriger les erreurs de détection du modèle.
                    Les corrections sont enregistrées automatiquement et peuvent être exportées.
                """)
                
                # Document selection for correction
                document_to_correct = st.selectbox(
                    "Sélectionner un document à corriger",
                    st.session_state.results_df['Nom_Document'].unique()
                )
                
                if document_to_correct:
                    # Create columns for layout
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Get the index of the selected document
                        row_index = st.session_state.results_df[
                            st.session_state.results_df['Nom_Document'] == document_to_correct
                        ].index[0]
                        
                        # Display the correction interface
                        self.display_correction_interface(
                            row_index,
                            st.session_state.results_df
                        )
                    
                    with col2:
                        # Show the original document content
                        if document_to_correct in st.session_state.processed_files:
                            st.markdown("### Document original")
                            file_content = st.session_state.processed_files[document_to_correct]
                            self.display_file_content(file_content, document_to_correct)
                
                # Download corrections log
                corrections_link = self.get_corrections_download_link()
                if corrections_link:
                    st.markdown("### Export des corrections")
                    st.markdown(corrections_link, unsafe_allow_html=True)
                
                # Display correction statistics
                if st.session_state.corrections:
                    with st.expander("📊 Statistiques des corrections", expanded=False):
                        corrections_by_entity = {}
                        for doc_corrs in st.session_state.corrections.values():
                            for corr in doc_corrs:
                                entity_type = corr['entity_type']
                                corrections_by_entity[entity_type] = \
                                    corrections_by_entity.get(entity_type, 0) + 1
                        
                        st.markdown("#### Corrections par type d'entité")
                        for entity, count in sorted(
                            corrections_by_entity.items(),
                            key=lambda x: x[1],
                            reverse=True
                        ):
                            st.markdown(f"- **{entity}**: {count} correction(s)")

if __name__ == "__main__":
    dashboard = MedicalAnnotationDashboard()
    dashboard.run()