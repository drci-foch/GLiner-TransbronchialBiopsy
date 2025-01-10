import streamlit as st
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import base64
import json
from config import config
from io import BytesIO  
from auth.user_auth import UserAuth

class UIComponents:
    """Manages all UI components for the application"""
    
    def __init__(self):
        """Initialize UI components"""
        self.config = config
        self.user_auth = UserAuth()

    def create_header(self):
        """Create application header"""
        with st.expander("📖 Instructions d'utilisation", expanded=False):
            st.markdown("""
                ### Comment utiliser FochAnnot :
                1. **Upload de fichiers** : Téléchargez vos documents (PDF ou TXT)
                2. **Extraction** : Le système extraira automatiquement la conclusion
                3. **Analyse** : Les entités seront détectées et structurées
                4. **Corrections** : Identifiez et corrigez les erreurs du modèle
                5. **Export** : Téléchargez les résultats et l'historique des corrections
            """)
    

    
    def create_sidebar(self, on_clear_results: Callable):
        """
        Create sidebar with controls.
        
        Args:
            on_clear_results: Callback for clearing results
        """
        with st.sidebar:
            if st.button("Se déconnecter"):
                self.user_auth.logout()
                st.rerun()


            st.header("📈 Informations")
            
            # Add file statistics - check if state exists and is updated
            if ('results_df' in st.session_state and 
                st.session_state.results_df is not None and 
                'processed_files' in st.session_state):
                
                # Main statistics - will update when processed_files changes
                total_files = len(st.session_state.processed_files)
                
                # Add statistics about conclusions
                if 'Conclusion' in st.session_state.results_df:
                    # Recalculate stats based on current state
                    docs_avec_conclusion = st.session_state.results_df['Conclusion'].notna().sum()
                    taux_conclusion = (docs_avec_conclusion / total_files) * 100 if total_files > 0 else 0
                    st.metric(label="Documents avec conclusion", value=f"{docs_avec_conclusion}/{total_files}")
                    st.progress(taux_conclusion/100, f"{taux_conclusion:.1f}%")

                # Label statistics with automatic updates
                if self.config.LABELS and not st.session_state.results_df.empty:
                    st.markdown("---")
                    st.markdown("### 🏷️ Étiquettes détectées")
                    for i in range(0, len(self.config.LABELS), 2):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if i < len(self.config.LABELS):
                                label = self.config.LABELS[i]
                                if label in st.session_state.results_df:
                                    label_count = st.session_state.results_df[label].notna().sum()
                                    label_taux = (label_count / total_files) * 100 if total_files > 0 else 0
                                    st.metric(label=label, value=f"{label_count}/{total_files}")
                                    st.progress(label_taux/100, f"{label_taux:.1f}%")
                        
                        with col2:
                            if i + 1 < len(self.config.LABELS):
                                label = self.config.LABELS[i + 1]
                                if label in st.session_state.results_df:
                                    label_count = st.session_state.results_df[label].notna().sum()
                                    label_taux = (label_count / total_files) * 100 if total_files > 0 else 0
                                    st.metric(label=label, value=f"{label_count}/{total_files}")
                                    st.progress(label_taux/100, f"{label_taux:.1f}%")
                
                # Actions - always available when there are files
                st.markdown("---")
                st.markdown("### 🗑️ Actions")

                if st.button("🗑️ Ré-initialiser les résultats", key="clear_button"):
                    on_clear_results()
                    st.experimental_rerun()

            
            # Version info
            st.markdown("---")
            st.markdown("### ℹ️ À propos")
            st.markdown("Version: BTB_structuration_1.0.0")
            st.markdown("Author: Sarra Ben Yahia")
        
    def create_file_uploader(self, on_clear_results: Callable) -> List[Any]:
        """
        Create file upload section with delete all button.
        
        Returns:
            List[Any]: List of uploaded files
        """
        # Initialize the key for file uploader
        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_files = st.file_uploader(
            "Choisissez un ou plusieurs fichiers (PDF ou TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Glissez-déposez vos fichiers ou cliquez pour les sélectionner",
            key=f"uploader_{st.session_state.uploader_key}"
        )

        return uploaded_files

    # And in your clear function:
    def _clear_results(self):
        """Clear all results and session state"""
        st.session_state.results_df = None
        st.session_state.processed_files = {}
        # Increment the key to force file uploader reset
        st.session_state.uploader_key += 1
    
    def create_results_tabs(
        self,
        results_df: pd.DataFrame,
        corrections: Dict,
        on_file_view: Callable,
        on_correction: Callable,
        charts: Dict[str, go.Figure]
    ):
        """
        Create tabs for results display.
        
        Args:
            results_df: Results dataframe
            corrections: Corrections data
            on_file_view: Callback for file viewing
            on_correction: Callback for corrections
            charts: Visualization charts
        """
        tab1, tab2, tab3 = st.tabs([
            "📊 Données structurées",
            "📈 Statistiques",
            "✏️ Corrections"
        ])
        
        with tab1:
            self._create_data_tab(results_df, on_file_view)
        
        with tab2:
            self._create_stats_tab(results_df, charts)
        
        with tab3:
            self._create_corrections_tab(results_df, corrections, on_correction)
    
    def _create_data_tab(
        self,
        results_df: pd.DataFrame,
        on_file_view: Callable
    ):
        """Create data display tab"""
        # Configure column display order
        column_order = ['Nom_Document', 'Date_Structuration', 'Conclusion']  # Add Conclusion
        column_order.extend(self.config.LABELS)
        column_order.append('Scores')
        
        # Display interactive dataframe
        st.dataframe(
            results_df[column_order],
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
                "Conclusion": st.column_config.TextColumn(
                    "Conclusion",
                    help="Texte de la conclusion extraite",
                    width="large",
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
    
    def _create_stats_tab(
        self,
        results_df: pd.DataFrame,
        charts: Dict[str, go.Figure]
    ):
        """Create statistics tab"""
        if len(results_df) > 0:
            col_stats1, col_stats2 = st.columns([1, 1])
            
            with col_stats1:
                st.markdown("### Distribution des entités par document")
                for filename in results_df['Nom_Document'].unique():
                    st.markdown(f"**Document : {filename}**")
                    doc_data = results_df[results_df['Nom_Document'] == filename]
                    entities_found = [
                        col for col in self.config.LABELS 
                        if doc_data[col].notna().any()
                    ]
                    
                    if entities_found:
                        for entity in entities_found:
                            value = doc_data[entity].iloc[0]
                            if pd.notna(value):
                                st.markdown(f"- {entity}: {value}")
                    else:
                        st.markdown("*Aucune entité détectée*")
                    st.markdown("---")
            
            with col_stats2:
                st.markdown("### Statistiques globales")
                total_docs = len(results_df)
                total_entities = sum(
                    results_df[label].notna().sum() 
                    for label in self.config.LABELS
                )
                
                st.metric("Documents analysés", total_docs)
                st.metric("Entités détectées", total_entities)
                
                if total_docs > 0:
                    st.metric(
                        "Moyenne d'entités par document",
                        f"{total_entities/total_docs:.1f}"
                    )
            
            # Display charts
            st.markdown("### Visualisations")
            for name, fig in charts.items():
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée disponible pour les statistiques")
    
    def _create_corrections_tab(
        self,
        results_df: pd.DataFrame,
        corrections: Dict,
        on_correction: Callable
    ):
        """Create corrections tab"""
        st.markdown("### Interface de correction")
        st.markdown("""
            Cette interface permet de corriger les erreurs de détection du modèle.
            Les corrections sont enregistrées automatiquement et peuvent être exportées.
        """)
        
        # Document selection
        document = st.selectbox(
            "Sélectionner un document à corriger",
            results_df['Nom_Document'].unique()
        )
        
        if document:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                self._create_correction_interface(
                    document,
                    results_df,
                    corrections,
                    on_correction
                )
            
            with col2:
                if document in corrections:
                    st.markdown("### État actuel du document")
                    latest_state = corrections[document]["latest_state"]
                    if latest_state:
                        st.markdown(f"**Dernière mise à jour:** {latest_state['last_updated']}")
                        for key, value in latest_state.items():
                            if key not in ['last_updated', 'Nom_Document']:
                                st.markdown(f"**{key}:** {value}")
                    
                    st.markdown("### Historique des corrections")
                    history = corrections[document]["history"]
                    if history:
                        for idx, correction in enumerate(history):
                            st.markdown(
                                f"**Correction {idx + 1}** - "
                                f"{datetime.fromisoformat(correction['timestamp']).strftime('%H:%M:%S')}"
                            )
                            st.markdown(f"- Entité: {correction['entity_type']}")
                            st.markdown(f"- Valeur originale: {correction['original_value']}")
                            st.markdown(f"- Valeur corrigée: {correction['corrected_value']}")
                            st.markdown("---")
                    else:
                        st.info("Aucune correction pour ce document")
    
    def _create_correction_interface(
        self,
        document: str,
        results_df: pd.DataFrame,
        corrections: Dict,
        on_correction: Callable
    ):
        """Create correction interface for a document"""
        doc_data = results_df[results_df['Nom_Document'] == document].iloc[0]
        
        entity_type = st.selectbox(
            "Sélectionner l'entité à corriger",
            self.config.LABELS
        )
        
        current_value = doc_data[entity_type]
        current_value = str(current_value) if pd.notna(current_value) else ""
        
        st.text_area(
            "Valeur actuelle",
            current_value,
            disabled=True
        )
        
        corrected_value = st.text_area(
            "Valeur corrigée"
        )
        
        if st.button("Soumettre la correction"):
            on_correction(document, entity_type, current_value, corrected_value)

    def create_corrections_tab(
        self,
        results_df: pd.DataFrame,
        corrections: Dict,
        on_correction: Callable
    ):
        """Create corrections tab"""
        st.markdown("### Interface de correction")
        st.markdown("""
            Cette interface permet de corriger les erreurs de détection du modèle.
            Les corrections sont enregistrées automatiquement et peuvent être exportées.
        """)
        
        # Document selection
        document = st.selectbox(
            "Sélectionner un document à corriger",
            results_df['Nom_Document'].unique()
        )
        
        if document:
            # Display the conclusion text first
            conclusion_text = results_df[
                results_df['Nom_Document'] == document
            ]['Conclusion'].iloc[0]
            
            with st.expander("📄 Conclusion du document", expanded=True):
                st.markdown("**Texte de la conclusion:**")
                st.text_area(
                    "",
                    conclusion_text,
                    height=150,
                    disabled=True,
                    key=f"conclusion_{document}"
                )
            
            # Rest of the correction interface
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Entity selection
                entity_type = st.selectbox(
                    "Sélectionner l'entité à corriger",
                    self.config.LABELS
                )
                
                # Get current value
                current_value = results_df[
                    results_df['Nom_Document'] == document
                ][entity_type].iloc[0]
                current_value = str(current_value) if pd.notna(current_value) else ""
                
                # Display current value
                st.text_area(
                    "Valeur actuelle",
                    current_value,
                    disabled=True,
                    key=f"current_{document}_{entity_type}"
                )
                
                # Input for correction
                corrected_value = st.text_area(
                    "Valeur corrigée",
                    key=f"correction_{document}_{entity_type}"
                )
                
                # Submit button
                if st.button("Soumettre la correction", key=f"submit_{document}_{entity_type}"):
                    on_correction(document, entity_type, current_value, corrected_value)
            
            with col2:
                # Display current document state
                if document in corrections:
                    st.markdown("### État actuel du document")
                    latest_state = corrections[document]["latest_state"]
                    if latest_state:
                        st.markdown(f"**Dernière mise à jour:** {latest_state['last_updated']}")
                        for key, value in latest_state.items():
                            if key not in ['last_updated', 'Nom_Document', 'Conclusion']:
                                st.markdown(f"**{key}:** {value}")
                    
                    st.markdown("### Historique des corrections")
                    history = corrections[document]["history"]
                    if history:
                        for idx, correction in enumerate(history):
                            st.markdown(
                                f"**Correction {idx + 1}** - "
                                f"{datetime.fromisoformat(correction['timestamp']).strftime('%H:%M:%S')}"
                            )
                            st.markdown(f"- Entité: {correction['entity_type']}")
                            st.markdown(f"- Valeur originale: {correction['original_value']}")
                            st.markdown(f"- Valeur corrigée: {correction['corrected_value']}")
                            st.markdown("---")
                    else:
                        st.info("Aucune correction pour ce document")
        
        # Add download button for corrections log
        if corrections:
            st.markdown("### Export des corrections")
            corrections_json = json.dumps(corrections, ensure_ascii=False, indent=2)
            b64 = base64.b64encode(corrections_json.encode()).decode()
            filename = f"corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="download-button">📥 Télécharger l\'historique des corrections</a>'
            st.markdown(href, unsafe_allow_html=True)

    def create_download_buttons(self, results_df: pd.DataFrame):
        """Create download buttons for results"""
        st.markdown("""
            <style>
            .download-section {
                background: linear-gradient(90deg, #00487E, #0079C0);
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: 20px 0;
                color: white;
            }
            .download-button-container {
                display: flex;
                gap: 20px;
                align-items: center;
                margin-top: 20px;
                padding: 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            .download-button-custom {
                background: white;
                color: #00487E;
                padding: 12px 24px;
                border-radius: 8px;
                text-decoration: none;
                display: inline-block;
                width: 100%;
                text-align: center;
                transition: transform 0.2s, box-shadow 0.2s;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                font-weight: 500;
            }
            .download-button-custom:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                background: #f8f9fa;
                text-decoration: none;
                color: #00487E;
            }
            .file-info {
                color: rgba(255, 255, 255, 0.9);
                font-size: 0.9em;
                margin-top: 5px;
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

        # CSV and Excel data preparation
        # CSV
        csv = results_df.to_csv(index=False)
        csv_size = len(csv.encode('utf-8')) / 1024  # Size in KB
        b64_csv = base64.b64encode(csv.encode()).decode()
        filename_csv = f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Résultats')
        excel_data = output.getvalue()
        excel_size = len(excel_data) / 1024  # Size in KB
        b64_excel = base64.b64encode(excel_data).decode()
        filename_excel = f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        # Create the complete HTML structure
        download_html = f"""
            <div class="download-section">
                <h3>📥 Export des résultats</h3>
                <div class="download-button-container">
                    <div style="flex: 1; text-align: center;">
                        <a href="data:file/csv;base64,{b64_csv}" 
                        download="{filename_csv}" 
                        class="download-button-custom">
                            <span>📊 Format CSV</span>
                        </a>
                        <div class="file-info">
                            Taille: {csv_size:.1f} KB
                        </div>
                    </div>
                    <div style="flex: 1; text-align: center;">
                        <a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" 
                        download="{filename_excel}" 
                        class="download-button-custom">
                            <span>📈 Format Excel</span>
                        </a>
                        <div class="file-info">
                            Taille: {excel_size:.1f} KB
                        </div>
                    </div>
                </div>
            </div>
        """

        st.markdown(download_html, unsafe_allow_html=True)

    def create_file_viewer(self, file_content: bytes, filename: str):
        """Create file viewer dialog"""
        with st.expander(f"📄 {filename}", expanded=True):
            col1, col2 = st.columns([0.9, 0.1])
            with col2:
                if st.button("❌", key=f"close_{filename}"):
                    st.rerun()
            
            file_type = filename.split('.')[-1].lower()
            
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
                st.text_area(
                    "",
                    file_content.decode('utf-8', errors='replace'),
                    height=800
                )
    
    def create_logs_viewer(self):
        """Create logs viewer interface"""
        st.markdown("### 📋 Historique des sessions")
        
        # Get all log files from the correction_logs directory
        log_files = sorted(
            Path("correction_logs").glob("*.json"),
            reverse=True
        )
        
        if not log_files:
            st.info("Aucun historique de session disponible")
            return
        
        # Create tabs for different sessions
        session_tabs = st.tabs([
            f"Session {datetime.strptime(f.stem.replace('corrections_log_', ''), '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')}"
            for f in log_files
        ])
        
        for tab, log_file in zip(session_tabs, log_files):
            with tab:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    if log_data:
                        # Count total documents and corrections
                        total_docs = len(log_data)
                        total_corrections = sum(
                            len(doc_data.get("history", []))
                            for doc_data in log_data.values()
                        )
                        
                        # Show session summary
                        st.markdown(f"**Documents traités:** {total_docs}")
                        st.markdown(f"**Corrections totales:** {total_corrections}")
                        
                        # Show details for each document
                        for doc_name, doc_data in log_data.items():
                            st.markdown(f"### 📄 {doc_name}")
                            
                            if "latest_state" in doc_data:
                                st.markdown("#### État final")
                                latest = doc_data["latest_state"]
                                for key, value in latest.items():
                                    if key != "last_updated":
                                        st.markdown(f"**{key}:** {value}")
                            
                            if "history" in doc_data:
                                st.markdown("#### Historique des corrections")
                                for corr in doc_data["history"]:
                                    st.markdown(
                                        f"- {corr['entity_type']}: "
                                        f"{corr['original_value']} → "
                                        f"{corr['corrected_value']} "
                                        f"({corr['timestamp']})"
                                    )
                            
                            st.markdown("---")
                    
                    # Add download button for this log
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        log_content = json.dumps(log_data, ensure_ascii=False, indent=2)
                        b64 = base64.b64encode(log_content.encode()).decode()
                        href = f'<a href="data:application/json;base64,{b64}" download="{log_file.name}" class="download-button">📥 Télécharger ce log</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la lecture du log: {str(e)}")



    def show_success(self, message: str):
        """Show success message"""
        st.success(message)
    
    def show_error(self, message: str):
        """Show error message"""
        st.error(message)
    
    def show_warning(self, message: str):
        """Show warning message"""
        st.warning(message)
    
    def show_info(self, message: str):
        """Show info message"""
        st.info(message)
