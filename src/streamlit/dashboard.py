import streamlit as st
import os
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import re
import pandas as pd
from typing import Dict, List
import plotly.graph_objects as go

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class MedicalAnnotationDashboard:
    def __init__(self):
        self.model = GLiNER.from_pretrained("../finetuning/models/custom_run/checkpoint-1000", load_tokenizer=True)

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

    def setup_page_config(self):
        """Configure the Streamlit page settings"""
        st.set_page_config(
            page_title="FochAnnot - Annotation Médicale",
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
                .stTextArea textarea {
                    font-size: 16px;
                    font-family: 'Arial', sans-serif;
                    line-height: 1.5;
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
                .stats-container {
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                }
                .legend-item {
                    display: flex;
                    align-items: center;
                    margin: 0.5rem 0;
                    padding: 0.5rem;
                    border-radius: 4px;
                    transition: all 0.2s ease;
                }
                .legend-item:hover {
                    transform: translateX(5px);
                }
            </style>
        """, unsafe_allow_html=True)

    def create_entity_stats(self, entities: List[Dict]) -> pd.DataFrame:
        """Create statistics from identified entities"""
        entity_counts = {}
        for entity in entities:
            label = entity["label"]
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        # Create DataFrame with all labels (including those with 0 count)
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

    def display_legend(self):
        """Display an interactive color legend"""
        st.markdown("### Légende des Entités")
        
        # Create two columns for the legend
        col1, col2 = st.columns(2)
        
        # Split labels into two groups
        mid_point = len(self.labels) // 2
        
        for i, (label, color) in enumerate(self.colors.items()):
            # Choose which column to place the legend item
            col = col1 if i < mid_point else col2
            
            col.markdown(
                f'''<div class="legend-item" 
                    style="background-color: {color}30; border-left: 4px solid {color}">
                    <span style="margin-left: 8px;">{label}</span>
                </div>''',
                unsafe_allow_html=True
            )

    def run(self):
        """Main application logic"""
        self.setup_page_config()
        self.apply_custom_css()

        # Header
        st.title("🏥 FochAnnot : Annotation des Données Médicales")

        with st.expander("📖 Instructions d'utilisation", expanded=False):
            st.markdown("""
                ### Comment utiliser FochAnnot :
                1. **Saisie du texte** : Entrez ou collez votre texte médical dans la zone de texte.
                2. **Ajustement du seuil** : Utilisez le curseur dans la barre latérale pour ajuster la sensibilité de la détection.
                3. **Visualisation** :
                   - Les entités détectées sont surlignées avec des couleurs distinctes
                   - Survolez une entité pour voir son type
                   - Consultez le graphique de distribution pour une vue d'ensemble
                4. **Légende** : Référez-vous à la légende pour identifier les types d'entités
            """)    
        st.markdown("---")

        # Sidebar for controls and stats
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

        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Texte à Analyser")
            input_text = st.text_area(
                "",
                height=300,
                value="""1/ Lavage broncho alvéolaire :
Liquide hypercellulaire avec légère polynucléose à polynucléaires neutrophiles sans agent pathogène
retrouvé.
2/ Biopsies transbronchiques : 7 fragments.
Absence de rejet aigu cellulaire bronchiolaire ou parenchymateux. A0 B0
Absence de lésions évocatrices de rejet aigu humoral.
Absence de lésions évocatrices de rejet chronique.
Absence d'inclusion virale et notamment d'inclusion de type CMV."""
            )

            if input_text:
                # Get predictions
                entities = self.model.predict_entities(input_text, self.labels, threshold=threshold)
                
                if entities:
                    st.subheader("Texte Annoté")
                    highlighted_text = self.highlight_entities(input_text, entities)
                    st.markdown(f'<div style="font-size:16px; line-height:1.6;">{highlighted_text}</div>',
                              unsafe_allow_html=True)
                else:
                    st.warning("Aucune entité n'a été identifiée dans le texte.")

        with col2:
            if input_text:
                st.subheader("Statistiques")
                stats_df = self.create_entity_stats(entities)
                
                # Display entity distribution chart
                st.plotly_chart(
                    self.create_entity_distribution_chart(stats_df),
                    use_container_width=True
                )
                
                # Display color legend
                self.display_legend()




if __name__ == "__main__":
    dashboard = MedicalAnnotationDashboard()
    dashboard.run()