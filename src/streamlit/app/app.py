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
from auth.user_auth import UserAuth
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import base64

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
        self.config = Config()
        self.user_auth = UserAuth()
 
        
        # Only initialize other components after user login
        self.model_handler = None
        self.text_processor = None
        self.file_handler = None
        self.entity_processor = None
        self.corrections_manager = None
        self.chart_generator = None
        self.ui = None
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = {}
        if 'results_df' not in st.session_state:
            st.session_state.results_df = None
        if 'corrections' not in st.session_state:
            st.session_state.corrections = {}

    def initialize_components(self):
        """Initialize components after user login"""
        if not self.model_handler:
            self.model_handler = ModelHandler()
            self.text_processor = TextProcessor()
            self.file_handler = FileHandler()
            self.entity_processor = EntityProcessor()
            self.corrections_manager = CorrectionsManager()
            self.chart_generator = ChartGenerator()
            self.ui = UIComponents()
            
            # Set user for corrections manager
            user = self.user_auth.get_current_user()
            if user:
                self.corrections_manager.set_user(user)

    def show_login(self):
        """Show login interface"""
        st.title("FochAnnot : Connexion")
        
        tab1, tab2 = st.tabs(["Connexion", "Inscription"])
        
        with tab1:
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            
            if st.button("Se connecter"):
                if self.user_auth.login(username, password):
                    st.success("Connexion réussie!")
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect")
        
        with tab2:
            new_username = st.text_input("Nouveau nom d'utilisateur")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            
            st.info("""
            ### Critères de sécurité pour le mot de passe :
            
            - **Au moins 8 caractères**
            - **Contient au moins une lettre majuscule**
            - **Contient au moins une lettre minuscule**
            - **Contient au moins un chiffre**
            - **Contient au moins un caractère spécial**
            
            Ces règles aident à garantir la sécurité de votre compte.
            """)
            if st.button("S'inscrire"):
                if new_password != confirm_password:
                    st.error("Les mots de passe ne correspondent pas")
                elif self.user_auth.register(new_username, new_password):
                    st.success("Inscription réussie! Vous pouvez maintenant vous connecter.")
                else:
                    st.error("Nom d'utilisateur déjà utilisé")


    def show_processing_status(self, total_files: int, processed_files: int, filename: str = None):
        """
        Show processing status with a floating message and progress bar.
        
        Args:
            total_files: Total number of files to process
            processed_files: Number of files processed so far
            filename: Name of the current file being processed
        """
        # Calculate progress
        progress = processed_files / total_files if total_files > 0 else 0
        
        # Create floating status message
        st.markdown("""
            <style>
            .floating-status {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: linear-gradient(90deg, #00487E, #0079C0);
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 999999;
                min-width: 200px;
                opacity: 0.95;
            }
            .status-progress {
                margin-top: 8px;
                background: rgba(255,255,255,0.2);
                border-radius: 4px;
                height: 4px;
                overflow: hidden;
            }
            .status-bar {
                height: 100%;
                background: #93BE1E;
                width: var(--progress);
                transition: width 0.3s ease;
            }
            </style>
            <div class="floating-status">
                <div>✅ {processed_files}/{total_files} documents traités</div>
                <div style="font-size: 0.8em; opacity: 0.8;">{filename if filename else ''}</div>
                <div class="status-progress">
                    <div class="status-bar" style="--progress: {progress * 100}%;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

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
            
            # self.ui.show_success(f"✅ {uploaded_file.name} traité avec succès")
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
            # Initialize session state variables
            self._initialize_session_state()
            
            st.set_page_config(
                page_title="FochAnnot",
                page_icon="🏥",
                layout="wide"
            )

            # Apply styles
            Styles.apply_all_styles()

            # Add fixed header HTML
            user = self.user_auth.get_current_user() if hasattr(self, 'user_auth') else None
            st.markdown(f"""
                <div class="fixed-header">
                    <div class="fixed-header-logo">
                        <img src="data:image/svg+xml;base64,{base64.b64encode(open('assets/logo.svg', 'rb').read()).decode()}" alt="Logo">
                        <span class="fixed-header-title">FochAnnot - Structuration automatique de documents</span>
                    </div>
                    <div class="fixed-header-user">
                        <div class="user-avatar">🧑‍🔬</div>
                        <span class="user-info">{user if user else 'Invité'}</span>
                    </div>
                </div>
                <div class="main-content">
                </div>
            """, unsafe_allow_html=True)

            # Check if user is logged in
            if not self.user_auth.get_current_user():
                self.show_login()
                return
            
            # Initialize components after login
            self.initialize_components()
            


            
            # Create header
            self.ui.create_header()
            
            # Create sidebar
            self.ui.create_sidebar(self._clear_results)
            
            # File upload
            uploaded_files = self.ui.create_file_uploader(self._clear_results)
            
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
                    
                    # Add a searchable select box for files
                    selected_file = st.selectbox(
                        "🔍 Rechercher un document",
                        options=list(st.session_state.processed_files.keys()),
                        index=None,
                        placeholder="Choisir un document..."
                    )
                    # Display content of selected file
                    if selected_file:
                        with st.expander(f"Document: {selected_file}", expanded=True):
                            # Check if file is PDF (assuming the file content is in bytes)
                            try:
                                # Display PDF viewer
                                base64_pdf = base64.b64encode(st.session_state.processed_files[selected_file]).decode('utf-8')
                                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)
                            except Exception as e:
                                st.error("Erreur lors de l'affichage du PDF")
                            
                            # Download button for PDF
                            st.download_button(
                                "📥 Télécharger",
                                st.session_state.processed_files[selected_file],
                                file_name=selected_file,
                                mime="application/pdf"
                            )
                    
                    # Optional: Keep the grid view as well
                    st.divider()  # Add a visual separator
                    
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
                # st.markdown("---")
                # self.ui.create_logs_viewer()
            
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
