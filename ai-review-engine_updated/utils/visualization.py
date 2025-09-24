"""
Visualization Module for Review Analysis Results
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import yaml

logger = logging.getLogger(__name__)


class ReviewVisualizer:
    """Class for creating visualizations of review analysis"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.colors = self.config['visualization']['color_scheme']
        self.chart_types = self.config['visualization']['chart_types']
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_sentiment_distribution_chart(self, sentiments: Dict) -> go.Figure:
        """
        Create pie chart showing sentiment distribution
        
        Args:
            sentiments: Dictionary with sentiment counts
            
        Returns:
            Plotly figure
        """
        labels = list(sentiments.keys())
        values = list(sentiments.values())
        colors = [self.colors.get(label, '#808080') for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo='label+percent',
            hole=0.3
        )])
        
        fig.update_layout(
            title="Overall Sentiment Distribution",
            title_font_size=20,
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_aspect_sentiment_chart(self, aspect_data: Dict) -> go.Figure:
        """
        Create stacked bar chart for aspect sentiments
        
        Args:
            aspect_data: Dictionary with aspect sentiment data
            
        Returns:
            Plotly figure
        """
        aspects = list(aspect_data.keys())
        positive = []
        negative = []
        neutral = []
        
        for aspect in aspects:
            data = aspect_data[aspect]
            total = data.get('total_mentions', 1)
            positive.append((data.get('positive', 0) / total) * 100)
            negative.append((data.get('negative', 0) / total) * 100)
            neutral.append((data.get('neutral', 0) / total) * 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Positive',
            x=aspects,
            y=positive,
            marker_color=self.colors['positive']
        ))
        
        fig.add_trace(go.Bar(
            name='Negative',
            x=aspects,
            y=negative,
            marker_color=self.colors['negative']
        ))
        
        fig.add_trace(go.Bar(
            name='Neutral',
            x=aspects,
            y=neutral,
            marker_color=self.colors['neutral']
        ))
        
        fig.update_layout(
            title="Sentiment by Aspect",
            xaxis_title="Aspects",
            yaxis_title="Percentage (%)",
            barmode='stack',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_aspect_radar_chart(self, aspect_data: Dict) -> go.Figure:
        """
        Create radar chart showing positive sentiment ratio per aspect
        
        Args:
            aspect_data: Dictionary with aspect sentiment data
            
        Returns:
            Plotly figure
        """
        aspects = list(aspect_data.keys())
        positive_ratios = []
        
        for aspect in aspects:
            data = aspect_data[aspect]
            total = data.get('total_mentions', 1)
            positive_ratio = (data.get('positive', 0) / total) * 100
            positive_ratios.append(positive_ratio)
        
        fig = go.Figure(data=go.Scatterpolar(
            r=positive_ratios,
            theta=aspects,
            fill='toself',
            marker=dict(color=self.colors['positive'])
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Aspect Performance Radar",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_temporal_sentiment_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create line chart showing sentiment trends over time
        
        Args:
            df: DataFrame with date and sentiment columns
            
        Returns:
            Plotly figure
        """
        # Ensure date column is datetime
        if 'date' not in df.columns:
            logger.warning("No date column found for temporal analysis")
            return go.Figure()
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and sentiment
        grouped = df.groupby([pd.Grouper(key='date', freq='D'), 'sentiment']).size().reset_index(name='count')
        
        fig = go.Figure()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = grouped[grouped['sentiment'] == sentiment]
            fig.add_trace(go.Scatter(
                x=sentiment_data['date'],
                y=sentiment_data['count'],
                mode='lines+markers',
                name=sentiment.capitalize(),
                line=dict(color=self.colors[sentiment])
            ))
        
        fig.update_layout(
            title="Sentiment Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Reviews",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_word_cloud(self, texts: List[str], sentiment: str = 'all') -> plt.Figure:
        """
        Create word cloud from review texts
        
        Args:
            texts: List of review texts
            sentiment: Filter for specific sentiment
            
        Returns:
            Matplotlib figure
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(combined_text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud - {sentiment.capitalize()} Reviews', fontsize=16)
        
        return fig
    
    def create_comparison_chart(self, products: List[Dict]) -> go.Figure:
        """
        Create comparison chart for multiple products
        
        Args:
            products: List of product analysis dictionaries
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Overall Ratings", "Aspect Comparison"),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Overall ratings bar chart
        product_names = [p['name'] for p in products]
        ratings = [p.get('overall_rating', 0) for p in products]
        
        fig.add_trace(
            go.Bar(x=product_names, y=ratings, marker_color='lightblue'),
            row=1, col=1
        )
        
        # Aspect comparison
        aspects = self.config['sentiment_analysis']['aspects'][:5]  # Top 5 aspects
        
        for product in products:
            aspect_scores = []
            for aspect in aspects:
                score = product.get('aspects', {}).get(aspect, {}).get('positive_ratio', 0) * 100
                aspect_scores.append(score)
            
            fig.add_trace(
                go.Scatter(
                    x=aspects,
                    y=aspect_scores,
                    mode='lines+markers',
                    name=product['name']
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=500, showlegend=True, title="Product Comparison")
        
        return fig
    
    def create_confidence_interval_chart(self, metrics: Dict) -> go.Figure:
        """
        Create chart showing model performance with confidence intervals
        
        Args:
            metrics: Dictionary with performance metrics and CIs
            
        Returns:
            Plotly figure
        """
        metric_names = list(metrics.keys())
        values = [m['value'] for m in metrics.values()]
        lower_bounds = [m['ci_lower'] for m in metrics.values()]
        upper_bounds = [m['ci_upper'] for m in metrics.values()]
        
        fig = go.Figure()
        
        # Add main values
        fig.add_trace(go.Scatter(
            x=metric_names,
            y=values,
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Value'
        ))
        
        # Add confidence intervals
        for i, metric in enumerate(metric_names):
            fig.add_shape(
                type="line",
                x0=i, x1=i,
                y0=lower_bounds[i], y1=upper_bounds[i],
                line=dict(color="gray", width=2)
            )
        
        fig.update_layout(
            title="Model Performance Metrics with 95% Confidence Intervals",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_summary_dashboard(self, analysis_results: Dict) -> Dict:
        """
        Create complete dashboard with all visualizations
        
        Args:
            analysis_results: Complete analysis results dictionary
            
        Returns:
            Dictionary of figures
        """
        dashboard = {}
        
        # Sentiment distribution
        if 'overall_sentiment_distribution' in analysis_results:
            dashboard['sentiment_distribution'] = self.create_sentiment_distribution_chart(
                analysis_results['overall_sentiment_distribution']
            )
        
        # Aspect sentiment chart
        if 'aspect_analysis' in analysis_results:
            dashboard['aspect_sentiment'] = self.create_aspect_sentiment_chart(
                analysis_results['aspect_analysis']
            )
            dashboard['aspect_radar'] = self.create_aspect_radar_chart(
                analysis_results['aspect_analysis']
            )
        
        # Key metrics summary
        dashboard['metrics'] = self._create_metrics_cards(analysis_results)
        
        return dashboard
    
    def _create_metrics_cards(self, analysis_results: Dict) -> go.Figure:
        """
        Create metric cards for key statistics
        
        Args:
            analysis_results: Analysis results dictionary
            
        Returns:
            Plotly figure with metric cards
        """
        fig = go.Figure()
        
        # Calculate key metrics
        total_reviews = analysis_results.get('total_reviews', 0)
        sentiment_dist = analysis_results.get('overall_sentiment_distribution', {})
        positive_pct = (sentiment_dist.get('positive', 0) / max(total_reviews, 1)) * 100
        
        # Create indicator charts
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_reviews,
                title={"text": "Total Reviews"},
                domain={'x': [0, 0.3], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=positive_pct,
                title={"text": "Positive %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.colors['positive']},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "lightgreen"}
                       ]},
                domain={'x': [0.35, 0.65], 'y': [0, 1]}
            ),
            row=1, col=2
        )
        
        # Top mentioned aspect
        if 'aspect_analysis' in analysis_results:
            top_aspect = max(
                analysis_results['aspect_analysis'].items(),
                key=lambda x: x[1].get('total_mentions', 0),
                default=("N/A", {})
            )[0]
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=len(analysis_results.get('key_insights', [])),
                    title={"text": f"Key Insights<br>Top: {top_aspect}"},
                    domain={'x': [0.7, 1], 'y': [0, 1]}
                ),
                row=1, col=3
            )
        
        fig.update_layout(height=250, showlegend=False)
        
        return fig
