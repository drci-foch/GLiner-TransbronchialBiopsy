import streamlit as st
from typing import Optional
import base64
from pathlib import Path
import json 
import datetime

class Styles:
    """Manages all styling for the application"""
    
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
                    color: #2c3e50;
                    padding-bottom: 1rem;
                    border-bottom: 2px solid #eee;
                    margin-bottom: 2rem;
                }
                
                /* Sidebar Styles */
                .css-1d391kg {  /* Sidebar */
                    background-color: #f8f9fa;
                    padding: 2rem 1rem;
                }
                
                .sidebar .sidebar-content {
                    background-color: #f8f9fa;
                }
                
                /* File Upload Styles */
                .uploadedFile {
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    padding: 1rem;
                    margin: 1rem 0;
                    border: 1px solid #e9ecef;
                }
                
                .stFileUploader {
                    padding: 2rem;
                    border: 2px dashed #ccc;
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
                    background-color: #f8f9fa;
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
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    cursor: pointer;
                    transition: background-color 0.3s ease;
                }
                
                .stButton>button:hover {
                    background-color: #45a049;
                }
                
                /* Download Button Styles */
                .download-button {
                    display: inline-block;
                    padding: 0.75rem 1.5rem;
                    background-color: #4CAF50;
                    color: white !important;
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.3s ease;
                    text-align: center;
                    margin: 1rem 0;
                }
                
                .download-button:hover {
                    background-color: #45a049;
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
                
                /* Tabs Styles */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 2rem;
                    border-bottom: 1px solid #dee2e6;
                }
                
                .stTabs [data-baseweb="tab"] {
                    padding: 1rem 2rem;
                    color: #6c757d;
                    font-weight: 500;
                }
                
                .stTabs [data-baseweb="tab"][aria-selected="true"] {
                    color: #2c3e50;
                    border-bottom: 2px solid #4CAF50;
                }
                
                /* File Viewer Styles */
                .document-viewer {
                    background-color: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                }
                
                .viewer-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 1rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid #dee2e6;
                }
                
                /* PDF Viewer Styles */
                iframe {
                    width: 100%;
                    height: 800px;
                    border: none;
                    border-radius: 4px;
                }
                
                /* Correction Interface Styles */
                .correction-form {
                    background-color: #f8f9fa;
                    padding: 2rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                }
                
                .correction-history {
                    background-color: #fff;
                    padding: 1rem;
                    border-radius: 4px;
                    border: 1px solid #dee2e6;
                    margin: 0.5rem 0;
                }
                
                /* Alert Styles */
                .stAlert {
                    padding: 1rem;
                    border-radius: 4px;
                    margin: 1rem 0;
                }
                
                /* Statistics Styles */
                .metric-container {
                    background-color: white;
                    padding: 1.5rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                }
                
                .metric-value {
                    font-size: 2rem;
                    font-weight: 600;
                    color: #2c3e50;
                }
                
                .metric-label {
                    font-size: 0.9rem;
                    color: #6c757d;
                    margin-top: 0.5rem;
                }
                
                /* Chart Styles */
                .chart-container {
                    background-color: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 2rem 0;
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
                    
                    iframe {
                        height: 400px;
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
                    --primary-color: #4CAF50;
                    --secondary-color: #2c3e50;
                    --background-color: #f8f9fa;
                    --text-color: #2c3e50;
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
        """Apply all styles"""
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
                        href = f'<a href="data:application/json;base64,{b64}" download="{log_file.name}" class="download-button">📥 Télécharger ce log</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du log: {str(e)}")
