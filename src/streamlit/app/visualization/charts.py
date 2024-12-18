import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from config import config
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChartGenerator:
    """Handles creation of various charts and visualizations"""
    
    def __init__(self):
        """Initialize chart generator with configuration"""
        self.colors = config.COLORS
        self.labels = config.LABELS
        
    def create_entity_distribution_chart(
        self,
        stats_df: pd.DataFrame,
        title: str = "Distribution des Entités Identifiées"
    ) -> go.Figure:
        """
        Create horizontal bar chart of entity distribution.
        
        Args:
            stats_df (pd.DataFrame): Statistics dataframe
            title (str): Chart title
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
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
                title=title,
                xaxis_title="Nombre d'occurrences",
                yaxis_title="Type d'entité",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating entity distribution chart: {str(e)}")
            raise
    
    def create_confidence_heatmap(
        self,
        data: pd.DataFrame,
        entity_scores: Dict[str, List[float]]
    ) -> go.Figure:
        """
        Create heatmap of entity confidence scores.
        
        Args:
            data (pd.DataFrame): Entity data
            entity_scores (Dict[str, List[float]]): Confidence scores
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Prepare data for heatmap
            documents = data['Nom_Document'].unique()
            score_matrix = []
            
            for doc in documents:
                doc_scores = []
                for label in self.labels:
                    scores = entity_scores.get((doc, label), [0])
                    doc_scores.append(np.mean(scores))
                score_matrix.append(doc_scores)
            
            fig = go.Figure(data=go.Heatmap(
                z=score_matrix,
                x=self.labels,
                y=documents,
                colorscale='Blues',
                hoverongaps=False,
                hovertemplate="Document: %{y}<br>" +
                             "Entity: %{x}<br>" +
                             "Score: %{z:.2f}<br>" +
                             "<extra></extra>"
            ))
            
            fig.update_layout(
                title="Scores de Confiance par Entité et Document",
                height=400 + (len(documents) * 20),
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Type d'entité",
                yaxis_title="Document"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating confidence heatmap: {str(e)}")
            raise
    
    def create_corrections_timeline(
        self,
        corrections: Dict[str, List[Dict]]
    ) -> go.Figure:
        """
        Create timeline of corrections.
        
        Args:
            corrections (Dict[str, List[Dict]]): Corrections data
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Prepare timeline data
            timeline_data = []
            for doc, doc_corrections in corrections.items():
                for correction in doc_corrections:
                    timeline_data.append({
                        'Document': doc,
                        'Entity_Type': correction['entity_type'],
                        'Timestamp': pd.to_datetime(correction['timestamp']),
                        'Original': correction['original_value'],
                        'Corrected': correction['corrected_value']
                    })
            
            if not timeline_data:
                return None
            
            df = pd.DataFrame(timeline_data)
            df = df.sort_values('Timestamp')
            
            fig = go.Figure()
            
            for entity_type in df['Entity_Type'].unique():
                entity_data = df[df['Entity_Type'] == entity_type]
                fig.add_trace(go.Scatter(
                    x=entity_data['Timestamp'],
                    y=[entity_type] * len(entity_data),
                    mode='markers',
                    name=entity_type,
                    marker=dict(
                        color=self.colors.get(entity_type, '#000000'),
                        size=10
                    ),
                    hovertemplate="Document: %{customdata[0]}<br>" +
                                "Original: %{customdata[1]}<br>" +
                                "Corrected: %{customdata[2]}<br>" +
                                "Time: %{x}<br>" +
                                "<extra></extra>",
                    customdata=entity_data[['Document', 'Original', 'Corrected']].values
                ))
            
            fig.update_layout(
                title="Chronologie des Corrections",
                xaxis_title="Date et Heure",
                yaxis_title="Type d'entité",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating corrections timeline: {str(e)}")
            raise
    
    def create_entity_relationship_graph(
        self,
        data: pd.DataFrame
    ) -> go.Figure:
        """
        Create network graph of entity relationships.
        
        Args:
            data (pd.DataFrame): Entity data
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Create entity co-occurrence matrix
            cooccurrence = np.zeros((len(self.labels), len(self.labels)))
            
            for _, row in data.iterrows():
                present_entities = [
                    label for label in self.labels 
                    if pd.notna(row[label])
                ]
                
                for i, entity1 in enumerate(present_entities):
                    for j, entity2 in enumerate(present_entities):
                        if i != j:
                            idx1 = self.labels.index(entity1)
                            idx2 = self.labels.index(entity2)
                            cooccurrence[idx1][idx2] += 1
            
            # Create network layout
            layout = self._create_circular_layout(len(self.labels))
            
            # Create edges
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for i in range(len(self.labels)):
                for j in range(len(self.labels)):
                    if cooccurrence[i][j] > 0:
                        edge_x.extend([layout[i][0], layout[j][0], None])
                        edge_y.extend([layout[i][1], layout[j][1], None])
                        edge_weights.append(cooccurrence[i][j])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    width=1,
                    color='rgba(160,160,160,0.5)'
                ),
                hoverinfo='none'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=[pos[0] for pos in layout],
                y=[pos[1] for pos in layout],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=[self.colors[label] for label in self.labels]
                ),
                text=self.labels,
                textposition="top center",
                hovertemplate="Entity: %{text}<br>" +
                             "<extra></extra>"
            ))
            
            fig.update_layout(
                title="Graphe des Relations entre Entités",
                showlegend=False,
                hovermode='closest',
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600,
                width=800
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating entity relationship graph: {str(e)}")
            raise
    
    def _create_circular_layout(self, n: int) -> List[Tuple[float, float]]:
        """
        Create circular layout for network graph.
        
        Args:
            n (int): Number of nodes
            
        Returns:
            List[Tuple[float, float]]: List of (x, y) coordinates
        """
        radius = 1
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        return [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    
    def create_correction_frequency_chart(
        self,
        corrections: Dict[str, List[Dict]]
    ) -> go.Figure:
        """
        Create chart showing correction frequency over time.
        
        Args:
            corrections (Dict[str, List[Dict]]): Corrections data
            
        Returns:
            go.Figure: Plotly figure object
        """
        try:
            # Prepare correction frequency data
            correction_times = []
            for doc_corrections in corrections.values():
                correction_times.extend([
                    pd.to_datetime(c['timestamp']) for c in doc_corrections
                ])
            
            if not correction_times:
                return None
            
            df = pd.DataFrame({'timestamp': correction_times})
            df['date'] = df['timestamp'].dt.date
            frequency = df['date'].value_counts().sort_index()
            
            fig = go.Figure(data=go.Scatter(
                x=frequency.index,
                y=frequency.values,
                mode='lines+markers',
                line=dict(color='rgba(0,100,180,0.8)'),
                marker=dict(size=8),
                hovertemplate="Date: %{x}<br>" +
                             "Corrections: %{y}<br>" +
                             "<extra></extra>"
            ))
            
            fig.update_layout(
                title="Fréquence des Corrections",
                xaxis_title="Date",
                yaxis_title="Nombre de corrections",
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correction frequency chart: {str(e)}")
            raise
    
    def create_dashboard(
        self,
        data: pd.DataFrame,
        corrections: Dict[str, List[Dict]],
        entity_scores: Dict[str, List[float]]
    ) -> Dict[str, go.Figure]:
        """
        Create complete dashboard with multiple charts.
        
        Args:
            data (pd.DataFrame): Entity data
            corrections (Dict[str, List[Dict]]): Corrections data
            entity_scores (Dict[str, List[float]]): Confidence scores
            
        Returns:
            Dict[str, go.Figure]: Dictionary of chart figures
        """
        try:
            dashboard = {}
            
            # Entity distribution
            stats_df = pd.DataFrame({
                'Entity': self.labels,
                'Count': [data[label].notna().sum() for label in self.labels]
            })
            dashboard['entity_distribution'] = self.create_entity_distribution_chart(stats_df)
            
            # Confidence heatmap
            dashboard['confidence_heatmap'] = self.create_confidence_heatmap(
                data,
                entity_scores
            )
            
            # Corrections timeline
            if corrections:
                dashboard['corrections_timeline'] = self.create_corrections_timeline(
                    corrections
                )
                dashboard['correction_frequency'] = self.create_correction_frequency_chart(
                    corrections
                )
            
            # Entity relationships
            dashboard['entity_relationships'] = self.create_entity_relationship_graph(
                data
            )
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'Nom_Document': ['doc1.pdf', 'doc2.pdf'],
        'Site': ['Site A', 'Site B'],
        'Grade A': ['A0', 'A1'],
        'Grade B': ['B0', 'B1']
    })
    
    sample_corrections = {
        'doc1.pdf': [
            {
                'entity_type': 'Grade A',
                'original_value': 'A0',
                'corrected_value': 'A1',
                'timestamp': '2024-01-01T10:00:00'
            }
        ]
    }
    
    sample_scores = {
        ('doc1.pdf', 'Grade A'): [0.95],
        ('doc1.pdf', 'Grade B'): [0.88]
    }
    
    # Create charts
    chart_generator = ChartGenerator()
    
    try:
        dashboard = chart_generator.create_dashboard(
            sample_data,
            sample_corrections,
            sample_scores
        )
        
        print("Dashboard charts created successfully:")
        for chart_name, figure in dashboard.items():
            print(f"- {chart_name}")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
