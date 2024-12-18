import streamlit as st
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
import base64
import json
from config import config
from io import BytesIO  # Add this import


class UIComponents:
    """Manages all UI components for the application"""
    
    def __init__(self):
        """Initialize UI components"""
        self.config = config
        
    def create_header(self):
        """Create application header"""
        st.title("🏥 FochAnnot : Analyse de Documents")
        
        with st.expander("📖 Instructions d'utilisation", expanded=False):
            st.markdown("""
                ### Comment utiliser FochAnnot :
                1. **Upload de fichiers** : Téléchargez vos documents (PDF ou TXT)
                2. **Extraction** : Le système extraira automatiquement la conclusion
                3. **Analyse** : Les entités seront détectées et structurées
                4. **Corrections** : Identifiez et corrigez les erreurs du modèle
                5. **Export** : Téléchargez les résultats et l'historique des corrections
            """)
    
    def create_sidebar(
        self,
        on_clear_results: Callable
    ):
        """
        Create sidebar with controls.
        
        Args:
            on_clear_results: Callback for clearing results
        """
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
            
            if 'results_df' in st.session_state and st.session_state.results_df is not None:
                st.markdown("### Actions")
                if st.button("🗑️ Effacer tous les résultats"):
                    on_clear_results()
            
            return threshold
    
    def create_file_uploader(self) -> List[Any]:
        """
        Create file upload section.
        
        Returns:
            List[Any]: List of uploaded files
        """
        return st.file_uploader(
            "Choisissez un ou plusieurs fichiers (PDF ou TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Glissez-déposez vos fichiers ou cliquez pour les sélectionner"
        )
    
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
        column_order = ['Nom_Document', 'Date_Structuration']
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
                "Scores": st.column_config.Column(
                    "Scores de confiance",
                    help="Scores de confiance pour chaque entité détectée",
                    width="medium",
                    disabled=True
                )
            },
            hide_index=True,
        )
        
        # Add view buttons for each file
        for idx, row in results_df.iterrows():
            filename = row['Nom_Document']
            if st.button(f"📄 Voir {filename}", key=f"view_{filename}"):
                on_file_view(filename)
    
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
                st.markdown("### Historique des corrections")
                if document in corrections:
                    for correction in corrections[document]:
                        with st.expander(
                            f"{correction['entity_type']} - "
                            f"{datetime.fromisoformat(correction['timestamp']).strftime('%Y-%m-%d %H:%M')}"
                        ):
                            st.markdown(f"**Original:** {correction['original_value']}")
                            st.markdown(f"**Corrigé:** {correction['corrected_value']}")
    
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
    
    def create_download_buttons(self, results_df: pd.DataFrame):
        """Create download buttons for results"""
        st.markdown("### Export des résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results_df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            filename_csv = f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="{filename_csv}" class="download-button">📥 Télécharger (CSV)</a>'
            st.markdown(href_csv, unsafe_allow_html=True)
        
        with col2:
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Résultats')
            
            b64_excel = base64.b64encode(output.getvalue()).decode()
            filename_excel = f"resultats_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="{filename_excel}" class="download-button">📥 Télécharger (Excel)</a>'
            st.markdown(href_excel, unsafe_allow_html=True)
    
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
