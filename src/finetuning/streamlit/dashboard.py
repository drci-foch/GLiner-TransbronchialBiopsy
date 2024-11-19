import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import re

# Charger le modèle GLiNER
trained_model = GLiNER.from_pretrained("../models/checkpoint-500", load_tokenizer=True)

# Labels pour la prédiction des entités
labels = [
    "Site", "Nombre Total De Fragments", "Nombre Total De Fragments Alvéolés", 
    "Grade A", "Grade B", "Rejet Chronique", "Coloration C4d", "Lésion Septale", 
    "Lésion Intra-Alvéolaire", "Éosinophilie", "Pneumonie Organisée", "DAD", 
    "Infection", "Autre Pathologie"
]

# Fonction pour surligner les entités avec des couleurs différentes
def highlight_entities(text, entities):
    # Définir des couleurs pour chaque entité
    colors = {
        "Site": "#A1D6A3",  # Soft green
        "Nombre Total De Fragments": "#8EC8F2",  # Soft blue
        "Nombre Total De Fragments Alvéolés": "#F9E26E",  # Light yellow
        "Grade A": "#F4A3A6",  # Soft coral
        "Grade B": "#F4C1D6",  # Soft pink
        "Rejet Chronique": "#F6D02F",  # Light yellow-orange
        "Coloration C4d": "#5EC7A2",  # Teal
        "Lésion Septale": "#5D9BCE",  # Light sky blue
        "Lésion Intra-Alvéolaire": "#5D8A96",  # Steel blue
        "Éosinophilie": "#B2EBF2",  # Light cyan
        "Pneumonie Organisée": "#D3D3D3",  # Light gray
        "DAD": "#F4B8D4",  # Light pink
        "Infection": "#FFF5BA",  # Pale yellow
        "Autre Pathologie": "#FFD2D2"  # Pale pink
    }

    # Remplacer chaque entité par le texte coloré avec un effet de survol (tooltip)
    for entity in entities:
        label = entity["label"]
        entity_text = entity["text"]
        color = colors.get(label, "#F5F5F5")  # Couleur par défaut si l'entité n'est pas dans le dictionnaire
        # Utilisation de balises HTML pour ajouter la couleur de fond et l'effet de survol (tooltip)
        text = re.sub(rf"({re.escape(entity_text)})", 
                      r'<span class="highlighted-entity" style="background-color: {}; padding: 0.2em; border-radius: 5px;" title="{}">\1</span>'.format(color, label), 
                      text)
    
    return text

# Fonction pour afficher la légende des couleurs
def display_legend():
    colors = {
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

    st.subheader("Légende des couleurs")
    legend_html = ""
    for label, color in colors.items():
        legend_html += f'<div style="background-color: {color}; padding: 10px; margin: 5px 0; border-radius: 8px; color: black; font-weight: bold;">{label}</div>'
    
    st.markdown(legend_html, unsafe_allow_html=True)

# Fonction principale de l'application Streamlit
def main():
    st.title("FochAnnot : Annotation des Données Médicales (GLiNER) - Biopsies Transbronchiques")
    
    # Ajouter une section d'information pour expliquer l'utilisation de l'application
    st.info("""
    **Instructions d'utilisation:**
    1. Entrez un texte dans la zone de texte ci-dessous.
    2. L'application identifiera et surlignera les entités spécifiques dans le texte, avec des couleurs distinctes pour chaque entité.
    3. Placez votre curseur sur une entité pour voir un **popup** indiquant son nom.
    4. Consultez la légende des couleurs ci-dessous pour voir à quelle entité chaque couleur correspond.
    """)

    # Champ d'entrée de texte
    input_text = st.text_area("Texte", height=300, value="""1/ Lavage broncho alvéolaire :
Liquide hypercellulaire avec légère polynucléose à polynucléaires neutrophiles sans agent pathogène
retrouvé.
2/ Biopsies transbronchiques : 7 fragments.
Absence de rejet aigu cellulaire bronchiolaire ou parenchymateux. A0 B0
Absence de lésions évocatrices de rejet aigu humoral.
Absence de lésions évocatrices de rejet chronique.
Absence d'inclusion virale et notamment d’inclusion de type CMV.""")

    # Prédiction des entités avec le modèle GLiNER
    entities = trained_model.predict_entities(input_text, labels, threshold=0.5)

    # Afficher les entités et leurs labels
    if entities:
        # Surligner les entités dans le texte
        highlighted_text = highlight_entities(input_text, entities)
        st.markdown(f'<div style="font-size:18px">{highlighted_text}</div>', unsafe_allow_html=True)
    else:
        st.write("Aucune entité identifiée.")
    
    # Afficher la légende
    display_legend()

    # Ajouter le CSS pour l'effet de survol (tooltip)
    st.markdown("""
    <style>
        .highlighted-entity {
            cursor: pointer;
            border-radius: 5px;
            display: inline-block;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

        .highlighted-entity:hover {
            background-color: #FFD700; /* Gold color on hover */
            transform: scale(1.1); /* Slightly increase size on hover */
        }

        .highlighted-entity[title]:hover::after {
            content: attr(title);
            position: absolute;
            background-color: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            top: -25px;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
