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
        """Extract conclusion section from the text"""
        patterns = [
            # Pattern for "C O N C L U S I O N" with spaced letters
            r"C\s*O\s*N\s*C\s*L\s*U\s*S\s*I\s*O\s*N\s*[\n\r]*([^C]+?)(?=(?:[A-Z][A-Z\s]{8,}|$))",
            # Original patterns as fallback
            r"(?i)CONCLUSION[\s:]+([^\n]+(?:\n(?![\r\n])[^\n]+)*)",
            r"(?i)CONCLUSION ET SYNTHESE[\s:]+([^\n]+(?:\n(?![\r\n])[^\n]+)*)",
            r"(?i)SYNTHESE[\s:]+([^\n]+(?:\n(?![\r\n])[^\n]+)*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                conclusion_text = match.group(1).strip()
                # Clean up the text
                conclusion_text = re.sub(r'\s+', ' ', conclusion_text)
                conclusion_text = conclusion_text.replace('Suresnes,', '')
                conclusion_text = conclusion_text.split('ADICAP')[0]  # Remove ADICAP codes
                conclusion_text = conclusion_text.split('Compte-rendu')[0]  # Remove signature line
                return conclusion_text.strip()
        return None

    def create_structured_data(self, entities, filename):
        """Convert entities to structured format with filename"""
        entity_dict = {
            'Nom_Document': filename,
            'Date_Structuration': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for label in self.labels:
            entity_dict[label] = None
            entity_dict[f"{label}_Score"] = None
            
        for entity in entities:
            label = entity['label']
            entity_dict[label] = entity['text']
            entity_dict[f"{label}_Score"] = round(entity['score'], 3)
            
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
                4. **Export** : Téléchargez les résultats au format Excel
                
                ### Fonctionnalités :
                - Support des fichiers PDF et TXT
                - Détection automatique des conclusions
                - Visualisation interactive des documents
                - Export des résultats en Excel
                - Analyse statistique des entités détectées
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

        # Main content area
    
        
        
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
            tab1, tab2 = st.tabs(["📊 Données structurées", "📈 Statistiques"])
            
            with tab1:
                # Display interactive dataframe
                st.dataframe(
                    st.session_state.results_df,
                    column_config={
                        "Nom_Document": st.column_config.Column(
                            "Document",
                            help="Cliquez pour voir le document",
                            width="medium",
                        ),
                        "Date_Analyse": st.column_config.DatetimeColumn(
                            "Date d'analyse",
                            help="Date et heure de l'analyse",
                            format="DD/MM/YYYY HH:mm",
                            width="medium",
                        )
                    },
                    hide_index=True,
                )
                
                # Add modal viewers for each file
                for idx, row in st.session_state.results_df.iterrows():
                    filename = row['Nom_Document']
                    if filename in st.session_state.processed_files:
                        file_content = st.session_state.processed_files[filename]
                        self.display_file_modal(file_content, filename)

                # Download button for Excel with proper styling
                st.markdown("### Export des résultats")
                st.markdown(self.get_excel_download_link(st.session_state.results_df), unsafe_allow_html=True)
            
            with tab2:
                if len(st.session_state.results_df) > 0:
                    # Create statistics and visualizations
                    col_stats1, col_stats2 = st.columns([1, 1])
                    
                    with col_stats1:
                        st.markdown("### Distribution des entités par document")
                        for filename in st.session_state.results_df['Nom_Document'].unique():
                            st.markdown(f"**Document : {filename}**")
                            doc_data = st.session_state.results_df[
                                st.session_state.results_df['Nom_Document'] == filename
                            ]
                            entities_found = [col for col in self.labels if doc_data[col].notna().any()]
                            if entities_found:
                                for entity in entities_found:
                                    score = doc_data[f"{entity}_Score"].iloc[0]
                                    st.markdown(
                                        f"- {entity}: {doc_data[entity].iloc[0]} "
                                        f"(score: {score:.3f})"
                                    )
                            else:
                                st.markdown("*Aucune entité détectée*")
                            st.markdown("---")
                    
                    with col_stats2:
                        st.markdown("### Statistiques globales")
                        total_docs = len(st.session_state.results_df)
                        total_entities = sum(
                            st.session_state.results_df[label].notna().sum()
                            for label in self.labels
                        )
                        
                        st.metric("Documents analysés", total_docs)
                        st.metric("Entités détectées", total_entities)
                        st.metric(
                            "Moyenne d'entités par document",
                            f"{total_entities/total_docs:.1f}"
                        )
                else:
                    st.info("Aucune donnée disponible pour les statistiques")

if __name__ == "__main__":
    dashboard = MedicalAnnotationDashboard()
    dashboard.run()