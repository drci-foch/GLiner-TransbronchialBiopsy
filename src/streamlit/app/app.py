import streamlit as st
from config import Config
from models.model_handler import ModelHandler
from utils.text_processor import TextProcessor
from utils.file_handler import FileHandler
from data.entity_processor import EntityProcessor
from data.corrections_manager import CorrectionsManager
from visualization.charts import ChartGenerator
from ui.components import UIComponents
from ui.styles import Styles
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MedicalAnnotationDashboard:
    """Main application class for medical document analysis dashboard"""
    
    def __init__(self):
        """Initialize the dashboard application"""
        # Load configuration
        self.config = Config()
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("correction_logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model_handler = ModelHandler()
        self.text_processor = TextProcessor()
        self.file_handler = FileHandler()
        self.entity_processor = EntityProcessor()
        self.corrections_manager = CorrectionsManager(base_log_dir="correction_logs")
        self.chart_generator = ChartGenerator()
        self.ui = UIComponents()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = {}
        if 'results_df' not in st.session_state:
            st.session_state.results_df = None
        if 'corrections' not in st.session_state:
            st.session_state.corrections = {}
    
    def process_file(self, uploaded_file) -> bool:
        """
        Process a single uploaded file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            bool: True if processing successful
        """
        try:
            # Read file content
            file_content = uploaded_file.read()
            file_type = self.file_handler.get_file_type(uploaded_file.name)
            
            if not file_type:
                self.ui.show_error(f"Type de fichier non supporté pour {uploaded_file.name}")
                return False
            
            # Validate file
            is_valid, error = self.file_handler.validate_file(uploaded_file, uploaded_file.name)
            if not is_valid:
                self.ui.show_error(f"Erreur de validation pour {uploaded_file.name}: {error}")
                return False
            
            # Store file content
            st.session_state.processed_files[uploaded_file.name] = file_content
            
            # Extract text
            file_text = self.text_processor.extract_text(file_content, file_type)
            if not file_text:
                self.ui.show_error(f"Impossible de lire {uploaded_file.name}")
                return False
            
            # Extract conclusion
            conclusion = self.text_processor.extract_conclusion(file_text)
            if not conclusion:
                self.ui.show_error(f"Pas de conclusion trouvée dans {uploaded_file.name}")
                return False
            
            # Predict entities
            threshold = st.session_state.get('confidence_threshold', 0.5)
            entities = self.model_handler.predict_entities(
                conclusion,
                self.config.LABELS,
                threshold
            )
            
            if not entities:
                self.ui.show_warning(f"Aucune entité identifiée dans {uploaded_file.name}")
                return False
            
            # Process entities - Pass the conclusion text here
            structured_data = self.entity_processor.process_entities(
                entities=entities,
                filename=uploaded_file.name,
                conclusion_text=conclusion  # Add this parameter
            )
            
            # Update results dataframe
            self._update_results_dataframe(structured_data)
            
            self.ui.show_success(f"✅ {uploaded_file.name} traité avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            self.ui.show_error(f"Erreur lors du traitement de {uploaded_file.name}")
            return False
        
    def _update_results_dataframe(self, new_data: Dict):
        """Update results dataframe with new data"""
        new_row = pd.DataFrame([new_data])
        if st.session_state.results_df is None:
            st.session_state.results_df = new_row
        else:
            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, new_row],
                ignore_index=True
            )
    
    def _handle_corrections(
        self,
        document: str,
        entity_type: str,
        original_value: str,
        corrected_value: str
    ):
        """Handle entity corrections"""
        if corrected_value != original_value:
            # Get the full row data
            row_data = st.session_state.results_df[
                st.session_state.results_df['Nom_Document'] == document
            ].iloc[0]
            
            # Add correction
            self.corrections_manager.add_correction(
                document=document,
                entity_type=entity_type,
                original_value=original_value,
                corrected_value=corrected_value,
                full_row_data=row_data
            )
            
            # Update results dataframe
            mask = st.session_state.results_df['Nom_Document'] == document
            st.session_state.results_df.loc[mask, entity_type] = corrected_value
            
            self.ui.show_success("Correction enregistrée!")
    
    def _clear_results(self):
        """Clear all results and processed files"""
        st.session_state.results_df = None
        st.session_state.processed_files = {}
        self.corrections_manager.clear_session()  # Clear and create new session
        st.rerun()
    
    def _generate_charts(self) -> Dict[str, Any]:
        """Generate visualization charts"""
        if st.session_state.results_df is not None:
            return self.chart_generator.create_dashboard(
                st.session_state.results_df,
                st.session_state.corrections,
                {}  # Entity scores would be added here
            )
        return {}
    
    def run(self):
        """Run the main application"""
        try:
            # Set page config
            st.set_page_config(
                page_title="FochAnnot - Analyse Documents",
                page_icon="🏥",
                layout="wide"
            )
            
            # Apply styles
            Styles.apply_all_styles()
            
            # Create header
            self.ui.create_header()
            
            # Create sidebar
            threshold = self.ui.create_sidebar(self._clear_results)
            st.session_state.confidence_threshold = threshold
            
            # File upload
            uploaded_files = self.ui.create_file_uploader()
            
            if uploaded_files:
                with st.spinner("Traitement des documents en cours..."):
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in st.session_state.processed_files:
                            self.process_file(uploaded_file)
            
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
                    # Configure column order and format
                    column_order = ['Nom_Document', 'Date_Structuration', 'Conclusion']
                    for label in self.config.LABELS:
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
                    
                    # Add modal viewers for each file
                    for filename in st.session_state.processed_files:
                        if st.button(f"📄 Voir {filename}", key=f"view_{filename}"):
                            self.ui.create_file_viewer(
                                st.session_state.processed_files[filename],
                                filename
                            )
                    
                    # Download buttons
                    self.ui.create_download_buttons(display_df)
                
                with tab2:
                    # Generate and display charts
                    charts = self.chart_generator.create_dashboard(
                        st.session_state.results_df,
                        self.corrections_manager.corrections,
                        {}  # Entity scores would be added here
                    )
                    
                    for name, fig in charts.items():
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Create correction interface
                    self.ui.create_corrections_tab(
                        st.session_state.results_df,
                        self.corrections_manager.corrections,
                        self._handle_corrections
                    )
                
                # Add session history viewer
                st.markdown("---")
                self.ui.create_logs_viewer()
                
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            self.ui.show_error("Une erreur est survenue dans l'application")
            raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    try:
        # Create and run dashboard
        dashboard = MedicalAnnotationDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        st.error("Une erreur fatale est survenue. Veuillez réessayer.")
