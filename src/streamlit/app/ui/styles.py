import streamlit as st
from typing import Optional
import base64
from pathlib import Path
import json 
from datetime import datetime

class Styles:
    """Manages all styling for the application"""
    
    @staticmethod
    def get_logo_html():
        """Get the SVG logo HTML"""
        logo_path = Path("assets/logo.svg")
        try:
            with open(logo_path, "r", encoding='utf-8') as f:
                svg_content = f.read()
            logo_html = f'''
                <div style="text-align: center; padding: 1rem;">
                    <div style="width: 400; margin: 0 auto;">
                        {svg_content}
            '''
            return logo_html
        except FileNotFoundError:
            return """
                <div style="text-align: center; padding: 1rem;">
                    <h1 style="color: #00487E;"> FochAnnot : Structuration automatique de documents </h1>
                </div>
            """

    @staticmethod
    def show_header():
        """Show the header with logo"""
        logo_html = Styles.get_logo_html()
        st.markdown("""
            <style>
                /* Header container styles */
                .header-container {
                    background: linear-gradient(to right, #00487E, #0079C0);
                    padding: 1rem;
                    border-radius: 0 0 10px 10px;
                    margin-bottom: 2rem;
                }
                
                /* Logo container styles */
                .logo-container {
                    text-align: center;
                    padding: 1rem;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 0 auto;
                    max-width: 300px;
                }
                
                /* SVG logo styles */
                .logo-container svg {
                    width: 100%;
                    height: auto;
                    max-width: 250px;
                }
            </style>
        """, unsafe_allow_html=True)
        st.markdown(logo_html, unsafe_allow_html=True)
    
    @staticmethod
    def apply_base_styles():
        """Apply base application styles"""
        st.markdown("""
            <style>
                /* Main Layout Styles */
                .main {
                    padding: 2rem;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                /* Header Styles */
                .stTitle {
                    font-family: 'Helvetica Neue', sans-serif;
                    color: #00487E;
                    padding-bottom: 1rem;
                    border-bottom: 2px solid #eee;
                    margin-bottom: 2rem;
                }
                
                /* Sidebar Styles */
                .css-1d391kg {  /* Sidebar */
                    background: linear-gradient(to bottom, #00487E, #0079C0);
                }
                
                .sidebar .sidebar-content {
                    background-color: #F2F2F2;
                }
                
                /* File Upload Styles */
                .uploadedFile {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 1rem 0;
                    border: 2px dashed #93BE1E;
                }
                
                .stFileUploader {
                    padding: 2rem;
                    border: 2px dashed #0079C0;
                    border-radius: 8px;
                    text-align: center;
                    background-color: #fcfcfc;
                    margin: 1rem 0;
                }
                
                /* Results Container Styles */
                .results-container {
                    background-color: #ffffff;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 2rem 0;
                }
                
                /* Table Styles */
                .dataframe {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1rem 0;
                }
                
                .dataframe th {
                    background-color: #00487E;
                    color: white;
                    padding: 0.75rem;
                    text-align: left;
                    font-weight: 600;
                    border-bottom: 2px solid #dee2e6;
                }
                
                .dataframe td {
                    padding: 0.75rem;
                    border-bottom: 1px solid #dee2e6;
                }
                
                /* Button Styles */
                .stButton>button {
                    background-color: #00487E;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                
                .stButton>button:hover {
                    background-color: #0079C0;
                }
                
                /* Download Button Styles */
                .download-button {
                    display: inline-block;
                    padding: 0.75rem 1.5rem;
                    background-color: #93BE1E;
                    color: white !important;
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.3s ease;
                    text-align: center;
                    margin: 1rem 0;
                }
                
                .download-button:hover {
                    background-color: #7da019;
                    text-decoration: none;
                }
                
                /* Entity Highlight Styles */
                .highlighted-entity {
                    cursor: help;
                    border-radius: 4px;
                    padding: 2px 4px;
                    display: inline-block;
                    transition: all 0.3s ease;
                    position: relative;
                    background-color: rgba(0, 72, 126, 0.1);
                }
                
                .highlighted-entity:hover {
                    transform: scale(1.05);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    background-color: rgba(0, 72, 126, 0.2);
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
                
                /* Tabs Styles */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 2rem;
                    border-bottom: 1px solid #dee2e6;
                }
                
                .stTabs [data-baseweb="tab"] {
                    padding: 1rem 2rem;
                    color: #00487E;
                    font-weight: 500;
                }
                
                .stTabs [data-baseweb="tab"][aria-selected="true"] {
                    color: white;
                    background: linear-gradient(to right, #0079C0, #93BE1E);
                    border-radius: 4px 4px 0 0;
                }
                
                /* File Viewer Styles */
                .document-viewer {
                    background-color: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                    border: 1px solid #00487E;
                }
                
                .viewer-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid #dee2e6;
                }
                
                /* Correction Interface Styles */
                .correction-form {
                    background-color: #f8f9fa;
                    padding: 2rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                    border: 1px solid #0079C0;
                }
                
                .correction-history {
                    background-color: #fff;
                    padding: 1rem;
                    border-radius: 4px;
                    border: 1px solid #93BE1E;
                    margin: 0.5rem 0;
                }
                
                /* Alert Styles */
                .stAlert {
                    padding: 1rem;
                    border-radius: 4px;
                    margin: 1rem 0;
                    border-left: 4px solid #00487E;
                }
                
                /* Statistics Styles */
                .metric-container {
                    background: linear-gradient(135deg, #00487E, #0079C0);
                    padding: 1.5rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                    color: white;
                }
                
                .metric-value {
                    font-size: 2rem;
                    font-weight: 600;
                    color: white;
                }
                
                .metric-label {
                    font-size: 0.9rem;
                    color: rgba(255, 255, 255, 0.9);
                    margin-top: 0.5rem;
                }
                
                /* Chart Styles */
                .chart-container {
                    background-color: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 2rem 0;
                    border: 1px solid #00487E;
                }
                
                /* Responsive Design */
                @media (max-width: 768px) {
                    .main {
                        padding: 1rem;
                    }
                    
                    .stFileUploader {
                        padding: 1rem;
                    }
                    
                    .results-container {
                        padding: 1rem;
                    }
                    
                    .document-viewer {
                        padding: 1rem;
                    }
                    
                    .stTabs [data-baseweb="tab"] {
                        padding: 0.5rem 1rem;
                    }
                }
                
                /* Print Styles */
                @media print {
                    .sidebar {
                        display: none !important;
                    }
                    
                    .stButton {
                        display: none !important;
                    }
                    
                    .download-button {
                        display: none !important;
                    }
                    
                    .document-viewer {
                        box-shadow: none;
                        border: 1px solid #dee2e6;
                    }
                }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def apply_custom_theme():
        """Apply custom theme colors"""
        st.markdown("""
            <style>
                :root {
                    --primary-color: #00487E;
                    --secondary-color: #0079C0;
                    --accent-color: #93BE1E;
                    --background-color: #F2F2F2;
                    --text-color: #00487E;
                    --border-color: #dee2e6;
                }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def apply_fonts():
        """Apply custom fonts"""
        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
                
                html, body, [class*="css"] {
                    font-family: 'Inter', sans-serif;
                }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def apply_animations():
        """Apply CSS animations"""
        st.markdown("""
            <style>
                /* Fade In Animation */
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                
                .results-container, .document-viewer, .chart-container {
                    animation: fadeIn 0.5s ease-in-out;
                }
                
                /* Slide In Animation */
                @keyframes slideIn {
                    from { transform: translateY(20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                
                .stAlert {
                    animation: slideIn 0.3s ease-out;
                }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def hide_streamlit_elements():
        """Hide default Streamlit elements"""
        hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                .viewerBadge_container__1QSob {display: none;}
                .stDeployButton {display: none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    @classmethod
    def apply_all_styles(cls):
        """Apply all styles and show header"""
        cls.apply_fixed_header()  # Add this line
        cls.show_header()
        cls.apply_base_styles()
        cls.apply_custom_theme()
        cls.apply_fonts()
        cls.apply_animations()
        cls.hide_streamlit_elements()

    def create_logs_viewer(self):
        """Create logs viewer interface"""
        with st.expander("📋 Historique des sessions", expanded=False):
            log_files = sorted(Path("correction_logs").glob("*.json"), reverse=True)
            
            for log_file in log_files:
                timestamp = log_file.stem.replace("corrections_log_", "")
                formatted_time = datetime.strptime(
                    timestamp,
                    '%Y%m%d_%H%M%S'
                ).strftime('%Y-%m-%d %H:%M:%S')
                
                with st.expander(f"Session du {formatted_time}"):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            log_data = json.load(f)
                        
                        # Display log data
                        st.json(log_data)
                        
                        # Add download button for this log
                        log_content = json.dumps(log_data, ensure_ascii=False, indent=2)
                        b64 = base64.b64encode(log_content.encode()).decode()
                        href = f'''
                            <a href="data:application/json;base64,{b64}" 
                               download="{log_file.name}" 
                               class="download-button"
                               style="background-color: #93BE1E; 
                                      display: inline-block;
                                      padding: 0.5rem 1rem;
                                      color: white;
                                      text-decoration: none;
                                      border-radius: 4px;
                                      margin: 0.5rem 0;">
                                📥 Télécharger ce log
                            </a>
                        '''
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du log: {str(e)}")
    
    @staticmethod
    def apply_logo_styles():
        """Apply specific styles for the logo"""
        st.markdown("""
            <style>
                .logo-wrapper {
                    background: linear-gradient(to right, #00487E, #0079C0);
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 2rem;
                }
                
                .logo-inner {
                    background: white;
                    padding: 1rem;
                    border-radius: 4px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                
                .logo-inner svg {
                    max-width: 250px;
                    height: auto;
                }
            </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_file_viewer_styles():
        """Create styles for file viewer"""
        return """
            <style>
                .file-viewer {
                    background-color: white;
                    border-radius: 8px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    border: 1px solid #00487E;
                }
                
                .file-viewer-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 1px solid #dee2e6;
                }
                
                .file-viewer-content {
                    max-height: 500px;
                    overflow-y: auto;
                    padding: 1rem;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                }
                
                .file-viewer-footer {
                    margin-top: 1rem;
                    padding-top: 0.5rem;
                    border-top: 1px solid #dee2e6;
                    display: flex;
                    justify-content: flex-end;
                }
            </style>
        """
    
    @staticmethod
    def get_theme_config():
        """Get theme configuration for config.toml"""
        return {
            "theme": {
                "primaryColor": "#00487E",
                "backgroundColor": "#F2F2F2",
                "secondaryBackgroundColor": "#F7F7F7",
                "textColor": "#00487E",
                "font": "sans-serif"
            }
        }
    

    @staticmethod
    def apply_fixed_header():
        """Apply fixed header bar styles"""
        st.markdown("""
            <style>
            /* Fixed header bar - increased z-index and adjusted stacking context */
            .fixed-header {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 70px;
                background: linear-gradient(90deg, #00487E, #0079C0);
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0 2rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                z-index: 999999 !important;  /* Increased z-index */
            }
            
            /* Adjust sidebar to stay below header */
            section[data-testid="stSidebar"] {
                position: relative;
                z-index: 99999 !important;
                margin-top: 70px !important;
                height: calc(100vh - 70px) !important;
            }
            
            /* Adjust the sidebar's internal elements */
            section[data-testid="stSidebar"] > div {
                height: calc(100vh - 70px);
            }
            
            /* Ensure main content stays below header */
            .main .block-container {
                padding-top: 90px !important;
                margin-top: 0px;
            }
            
            /* Logo in fixed header */
            .fixed-header-logo {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            
            .fixed-header-logo img {
                height: 40px;
                width: auto;
            }
            
            .fixed-header-title {
                color: white;
                font-size: 1.5rem;
                font-weight: 600;
            }
            
            /* User info in header */
            .fixed-header-user {
                display: flex;
                align-items: center;
                gap: 1rem;
                color: white;
            }
            
            .user-avatar {
                width: 50px;
                height: 50px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .fixed-header {
                    padding: 0 1rem;
                }
                
                .fixed-header-title {
                    font-size: 1.2rem;
                }
                
                .user-info {
                    display: none;
                }
            }
            </style>
        """, unsafe_allow_html=True)



def create_config_toml():
    """Create .streamlit/config.toml file with theme settings"""
    config_dir = Path(".streamlit")
    config_file = config_dir / "config.toml"
    
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
    
    config_content = """
[theme]
primaryColor = "#00487E"
backgroundColor = "#F2F2F2"
secondaryBackgroundColor = "#F7F7F7"
textColor = "#00487E"
font = "sans-serif"

[server]
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
    """
    
    with open(config_file, "w") as f:
        f.write(config_content.strip())
