"""
Advanced Visualization Module for Quick Decision Making
Creates intuitive charts and graphs for phone review analysis
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import colorsys

class DecisionVisualizer:
    """Creates comprehensive visualizations for quick decision making"""
    
    def __init__(self):
        # Color schemes for consistent branding
        self.colors = {
            'positive': '#00C853',  # Green
            'negative': '#FF3D00',  # Red
            'neutral': '#FFB300',   # Amber
            'primary': '#1E88E5',   # Blue
            'secondary': '#7C4DFF', # Purple
            'warning': '#FF9800',   # Orange
            'success': '#4CAF50',   # Light Green
            'danger': '#F44336',    # Light Red
            'info': '#2196F3',      # Light Blue
            'dark': '#424242',      # Dark Gray
            'light': '#F5F5F5'      # Light Gray
        }
        
        # Chart templates
        self.templates = {
            'clean': dict(
                layout=go.Layout(
                    font=dict(family="Segoe UI, Arial", size=12),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=60, b=40),
                    hovermode='x unified'
                )
            )
        }
    
    def create_decision_dashboard(:
        self,
        phone_model: str,
        analysis_result: Dict[str, Any],
        confidence_level: float
    ) -> Dict[str, go.Figure]:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Returns:
            Dictionary of Plotly figures for different aspects
        """
        
        figures = {}
        
        # 1. Main Decision Score Card
        figures['decision_score'] = self.create_decision_score_card(
            phone_model, analysis_result, confidence_level
        )
        
        # 2. Sentiment Gauge
        figures['sentiment_gauge'] = self.create_sentiment_gauge(
            analysis_result.get('sentiment', {})
        )
        
        # 3. Aspect Radar Chart
        figures['aspect_radar'] = self.create_aspect_radar_chart(
            analysis_result.get('aspects', {})
        )
        
        # 4. Confidence Indicator
        figures['confidence_meter'] = self.create_confidence_meter(
            confidence_level,
            analysis_result.get('data_quality', 'unknown')
        )
        
        # 5. Quick Decision Matrix
        figures['decision_matrix'] = self.create_decision_matrix(
            analysis_result
        )
        
        # 6. Pros vs Cons Balance
        figures['pros_cons'] = self.create_pros_cons_chart(
            analysis_result
        )
        
        # 7. Comparison Benchmark
        figures['benchmark'] = self.create_benchmark_chart(
            phone_model, analysis_result
        )
        
        # 8. Trend Prediction
        figures['trend'] = self.create_trend_chart(
            analysis_result
        )
        
        return figures
    
    def create_decision_score_card(:
        self,
        phone_model: str,
        analysis_result: Dict,
        confidence: float
    ) -> go.Figure:
        """Create a comprehensive decision score card"""
        
        # Calculate overall decision score
        sentiment = analysis_result.get('sentiment', {})
        positive_pct = sentiment.get('positive', 0)
        negative_pct = sentiment.get('negative', 0)
        
        # Decision score formula
        decision_score = (positive_pct - negative_pct + 100) / 2
        decision_score = min(max(decision_score, 0), 100)
        
        # Determine recommendation
        if decision_score >= 80:
            recommendation = "HIGHLY RECOMMENDED"
            rec_color = self.colors['success']
            emoji = "✅"
        elif decision_score >= 65:
            recommendation = "RECOMMENDED"
            rec_color = self.colors['primary']
            emoji = "👍"
        elif decision_score >= 50:
            recommendation = "CONSIDER CAREFULLY"
            rec_color = self.colors['warning']
            emoji = "🤔"
        else:
            recommendation = "NOT RECOMMENDED"
            rec_color = self.colors['danger']
            emoji = "⚠️"
        
        # Adjust for confidence
        if confidence < 0.4:
            recommendation += " (Low Confidence)"
            
        # Create the figure
        fig = go.Figure()
        
        # Add main score indicator
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=decision_score,
            title={'text': f"<b>{phone_model}</b><br>Decision Score", 'font': {'size': 24}},
            delta={'reference': 65, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': rec_color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 50], 'color': '#ffebee'},
                    {'range': [50, 65], 'color': '#fff3e0'},
                    {'range': [65, 80], 'color': '#e3f2fd'},
                    {'range': [80, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': decision_score
                }
            }
        ))
        
        # Add recommendation text
        fig.add_annotation(
            text=f"<b>{emoji} {recommendation}</b>",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=18, color=rec_color),
            xanchor='center'
        )
        
        # Add confidence note
        conf_text = f"Confidence: {self._get_confidence_label(confidence)}"
        fig.add_annotation(
            text=conf_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.25,
            showarrow=False,
            font=dict(size=12, color='gray'),
            xanchor='center'
        )
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=100, b=80),
            paper_bgcolor='white',
            font={'family': "Arial"}
        )
        
        return fig
    
    def create_sentiment_gauge(self, sentiment: Dict) -> go.Figure:
        """Create an intuitive sentiment gauge"""
        
        positive = sentiment.get('positive', 0)
        negative = sentiment.get('negative', 0)
        neutral = sentiment.get('neutral', 0)
        
        # Create subplot with 3 gauges
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            column_widths=[0.33, 0.33, 0.33]
        )
        
        # Positive gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=positive,
            title={'text': "😊 Positive", 'font': {'size': 16, 'color': self.colors['positive']}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': self.colors['positive']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': self.colors['positive'],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ), row=1, col=1)
        
        # Neutral gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=neutral,
            title={'text': "😐 Neutral", 'font': {'size': 16, 'color': self.colors['neutral']}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': self.colors['neutral']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': self.colors['neutral']
            }
        ), row=1, col=2)
        
        # Negative gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=negative,
            title={'text': "😔 Negative", 'font': {'size': 16, 'color': self.colors['negative']}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': self.colors['negative']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': self.colors['negative']
            }
        ), row=1, col=3)
        
        fig.update_layout(
            title="<b>Sentiment Analysis</b>",
            title_x=0.5,
            height=250,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
    
    def create_aspect_radar_chart(self, aspects: Dict) -> go.Figure:
        """Create a radar chart for aspect analysis"""
        
        # Extract aspect details
        aspect_details = aspects.get('details', {})
        
        if not aspect_details:
            # Default aspects if none available
            aspect_details = {
                'camera': {'sentiment': {'positive': 70}, 'is_estimated': True},
                'battery': {'sentiment': {'positive': 65}, 'is_estimated': True},
                'performance': {'sentiment': {'positive': 75}, 'is_estimated': True},
                'display': {'sentiment': {'positive': 80}, 'is_estimated': True},
                'price': {'sentiment': {'positive': 60}, 'is_estimated': True},
                'design': {'sentiment': {'positive': 70}, 'is_estimated': True}
            }
        
        categories = []
        values = []
        colors = []
        
        for aspect, data in aspect_details.items():
            categories.append(aspect.capitalize())
            sentiment = data.get('sentiment', {})
            positive_score = sentiment.get('positive', 50)
            values.append(positive_score)
            
            # Color based on score
            if positive_score >= 70:
                colors.append(self.colors['success'])
            elif positive_score >= 50:
                colors.append(self.colors['warning'])
            else:
                colors.append(self.colors['danger'])
        
        # Close the radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        fig = go.Figure()
        
        # Add the radar trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(30, 136, 229, 0.2)',
            line=dict(color=self.colors['primary'], width=2),
            marker=dict(size=8, color=self.colors['primary']),
            text=[f"{v:.0f}%" for v in values],
            hovertemplate='%{theta}: %{r:.0f}% positive<extra></extra>'
        ))
        
        # Add reference line at 50%
        fig.add_trace(go.Scatterpolar(
            r=[50] * len(categories),
            theta=categories,
            mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="<b>Feature Satisfaction Radar</b>",
            title_x=0.5,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%',
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=False,
            height=400,
            margin=dict(l=80, r=80, t=100, b=80),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_confidence_meter(self, confidence: float, data_quality: str) -> go.Figure:
        """Create a confidence level indicator"""
        
        confidence_pct = confidence * 100
        
        # Determine color based on confidence
        if confidence >= 0.75:
            color = self.colors['success']
            label = "High Confidence"
        elif confidence >= 0.5:
            color = self.colors['primary']
            label = "Moderate Confidence"
        elif confidence >= 0.3:
            color = self.colors['warning']
            label = "Low Confidence"
        else:
            color = self.colors['danger']
            label = "Very Low Confidence"
        
        fig = go.Figure()
        
        # Add the confidence bar
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            title={'text': f"<b>Analysis Confidence</b><br><span style='font-size:14px'>{label}</span>"},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color, 'thickness': 0.7},
                'bgcolor': "lightgray",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#ffebee'},
                    {'range': [30, 50], 'color': '#fff3e0'},
                    {'range': [50, 75], 'color': '#e3f2fd'},
                    {'range': [75, 100], 'color': '#e8f5e9'}
                ]
            }
        ))
        
        # Add data quality note
        quality_emoji = {
            'high': '📊', 'medium': '📈', 'low': '📉',
            'insufficient': '⚠️', 'no_data': '❌', 'unknown': '❓'
        }
        
        fig.add_annotation(
            text=f"{quality_emoji.get(data_quality, '❓')} Data Quality: {data_quality.upper()}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color='gray'),
            xanchor='center'
        )
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=80, b=60),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_decision_matrix(self, analysis: Dict) -> go.Figure:
        """Create a decision matrix showing pros/cons balance"""
        
        sentiment = analysis.get('sentiment', {})
        aspects = analysis.get('aspects', {}).get('details', {})
        
        # Calculate scores for different decision factors
        factors = {
            'Overall Satisfaction': sentiment.get('positive', 50),
            'Value for Money': aspects.get('price', {}).get('sentiment', {}).get('positive', 50),
            'Performance': aspects.get('performance', {}).get('sentiment', {}).get('positive', 60),
            'Reliability': 100 - sentiment.get('negative', 20),
            'Features': aspects.get('camera', {}).get('sentiment', {}).get('positive', 60),
            'User Experience': (sentiment.get('positive', 50) + 100 - sentiment.get('negative', 20)) / 2
        }
        
        # Create bar chart
        fig = go.Figure()
        
        # Sort factors by score
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        names = [f[0] for f in sorted_factors]
        values = [f[1] for f in sorted_factors]
        
        # Color bars based on value
        colors = []
        for v in values:
            if v >= 70:
                colors.append(self.colors['success'])
            elif v >= 50:
                colors.append(self.colors['primary'])
            else:
                colors.append(self.colors['warning'])
        
        fig.add_trace(go.Bar(
            y=names,
            x=values,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.0f}%' for v in values],
            textposition='auto',
            hovertemplate='%{y}: %{x:.0f}%<extra></extra>'
        ))
        
        # Add reference line at 50%
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add annotations for decision zones
        fig.add_annotation(
            text="👎 Poor", x=25, y=-0.5,
            showarrow=False, font=dict(color='red', size=10)
        )
        fig.add_annotation(
            text="👍 Good", x=75, y=-0.5,
            showarrow=False, font=dict(color='green', size=10)
        )
        
        fig.update_layout(
            title="<b>Decision Factors Matrix</b>",
            title_x=0.5,
            xaxis_title="Score (%)",
            xaxis=dict(range=[0, 100]),
            height=300,
            margin=dict(l=120, r=20, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        return fig
    
    def create_pros_cons_chart(self, analysis: Dict) -> go.Figure:
        """Create a balanced pros vs cons visualization"""
        
        aspects = analysis.get('aspects', {}).get('details', {})
        
        pros = []
        cons = []
        
        for aspect, data in aspects.items():
            sentiment = data.get('sentiment', {})
            if sentiment.get('positive', 0) > 60:
                pros.append({
                    'aspect': aspect.capitalize(),
                    'score': sentiment.get('positive', 0)
                })
            elif sentiment.get('negative', 0) > 30:
                cons.append({
                    'aspect': aspect.capitalize(),
                    'score': sentiment.get('negative', 0)
                })
        
        # Sort by score
        pros = sorted(pros, key=lambda x: x['score'], reverse=True)[:4]
        cons = sorted(cons, key=lambda x: x['score'], reverse=True)[:4]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("✅ Strengths", "⚠️ Weaknesses"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add pros
        if pros:
            fig.add_trace(
                go.Bar(
                    y=[p['aspect'] for p in pros],
                    x=[p['score'] for p in pros],
                    orientation='h',
                    marker_color=self.colors['success'],
                    text=[f"{p['score']:.0f}%" for p in pros],
                    textposition='auto',
                    showlegend=False,
                    hovertemplate='%{y}: %{x:.0f}% positive<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add cons
        if cons:
            fig.add_trace(
                go.Bar(
                    y=[c['aspect'] for c in cons],
                    x=[c['score'] for c in cons],
                    orientation='h',
                    marker_color=self.colors['danger'],
                    text=[f"{c['score']:.0f}%" for c in cons],
                    textposition='auto',
                    showlegend=False,
                    hovertemplate='%{y}: %{x:.0f}% negative<extra></extra>'
                ),
                row=1, col=2
            )
        
        fig.update_xaxes(range=[0, 100], row=1, col=1)
        fig.update_xaxes(range=[0, 100], row=1, col=2)
        
        fig.update_layout(
            title="<b>Pros vs Cons Analysis</b>",
            title_x=0.5,
            height=300,
            margin=dict(l=20, r=20, t=80, b=20),
            paper_bgcolor='white',
            showlegend=False
        )
        
        return fig
    
    def create_benchmark_chart(self, phone_model: str, analysis: Dict) -> go.Figure:
        """Create a benchmark comparison against category average"""
        
        # Simulate benchmark data (in production, this would come from database)
        category = self._determine_category(phone_model)
        
        benchmarks = {
            'flagship': {'satisfaction': 72, 'value': 65, 'performance': 85, 'camera': 80},
            'mid-range': {'satisfaction': 65, 'value': 75, 'performance': 70, 'camera': 65},
            'budget': {'satisfaction': 60, 'value': 85, 'performance': 60, 'camera': 55}
        }
        
        benchmark = benchmarks.get(category, benchmarks['mid-range'])
        
        # Get actual scores
        sentiment = analysis.get('sentiment', {})
        aspects = analysis.get('aspects', {}).get('details', {})
        
        actual = {
            'satisfaction': sentiment.get('positive', 50),
            'value': aspects.get('price', {}).get('sentiment', {}).get('positive', 50),
            'performance': aspects.get('performance', {}).get('sentiment', {}).get('positive', 60),
            'camera': aspects.get('camera', {}).get('sentiment', {}).get('positive', 60)
        }
        
        categories = list(benchmark.keys())
        
        fig = go.Figure()
        
        # Add benchmark trace
        fig.add_trace(go.Scatterpolar(
            r=[benchmark[cat] for cat in categories],
            theta=[cat.capitalize() for cat in categories],
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.1)',
            line=dict(color='gray', width=2, dash='dash'),
            name=f'{category.capitalize()} Average'
        ))
        
        # Add actual trace
        fig.add_trace(go.Scatterpolar(
            r=[actual[cat] for cat in categories],
            theta=[cat.capitalize() for cat in categories],
            fill='toself',
            fillcolor='rgba(30, 136, 229, 0.2)',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=10, color=self.colors['primary']),
            name=phone_model
        ))
        
        fig.update_layout(
            title=f"<b>Performance vs {category.capitalize()} Average</b>",
            title_x=0.5,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            showlegend=True,
            legend=dict(x=0, y=1),
            height=350,
            margin=dict(l=80, r=80, t=100, b=80),
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_trend_chart(self, analysis: Dict) -> go.Figure:
        """Create a simulated trend chart showing sentiment over time"""
        
        # Simulate trend data (in production, this would be real historical data)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Generate realistic trend based on current sentiment
        current_positive = analysis.get('sentiment', {}).get('positive', 65)
        
        # Add some noise and trend
        trend = np.linspace(current_positive - 10, current_positive, 30)
        noise = np.random.normal(0, 3, 30)
        positive_trend = np.clip(trend + noise, 0, 100)
        
        # Generate other sentiments
        negative_trend = np.clip(100 - positive_trend - 15 + np.random.normal(0, 2, 30), 0, 30)
        neutral_trend = 100 - positive_trend - negative_trend
        
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=dates, y=positive_trend,
            mode='lines+markers',
            name='Positive',
            line=dict(color=self.colors['positive'], width=2),
            fill='tonexty',
            fillcolor='rgba(0, 200, 83, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=neutral_trend,
            mode='lines+markers',
            name='Neutral',
            line=dict(color=self.colors['neutral'], width=2),
            fill='tonexty',
            fillcolor='rgba(255, 179, 0, 0.1)'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=negative_trend,
            mode='lines+markers',
            name='Negative',
            line=dict(color=self.colors['negative'], width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 61, 0, 0.1)'
        ))
        
        # Add trend indicator
        if positive_trend[-1] > positive_trend[0]:
            trend_text = "📈 Improving"
            trend_color = self.colors['success']
        elif positive_trend[-1] < positive_trend[0] - 5:
            trend_text = "📉 Declining"
            trend_color = self.colors['danger']
        else:
            trend_text = "➡️ Stable"
            trend_color = self.colors['primary']
        
        fig.add_annotation(
            text=f"<b>Trend: {trend_text}</b>",
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            showarrow=False,
            font=dict(size=14, color=trend_color),
            xanchor='right',
            bgcolor='white',
            bordercolor=trend_color,
            borderwidth=1
        )
        
        fig.update_layout(
            title="<b>Sentiment Trend (Last 30 Days)</b>",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Percentage (%)",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            height=300,
            margin=dict(l=60, r=20, t=60, b=40),
            paper_bgcolor='white',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(x=0, y=1)
        )
        
        return fig
    
    def create_quick_decision_summary(self, analysis: Dict) -> go.Figure:
        """Create a single comprehensive summary chart for quick decision"""
        
        # Calculate key metrics
        sentiment = analysis.get('sentiment', {})
        decision_score = (sentiment.get('positive', 0) - sentiment.get('negative', 0) + 100) / 2
        
        # Determine buy/wait/skip recommendation
        if decision_score >= 70:
            decision = "BUY"
            decision_color = self.colors['success']
            decision_emoji = "✅"
        elif decision_score >= 50:
            decision = "WAIT"
            decision_color = self.colors['warning']
            decision_emoji = "⏰"
        else:
            decision = "SKIP"
            decision_color = self.colors['danger']
            decision_emoji = "❌"
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Decision", "Sentiment", "Key Metrics", "Confidence"),
            specs=[
                [{"type": "indicator"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # 1. Decision indicator
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=decision_score,
            title={"text": f"{decision_emoji} {decision}"},
            number={"suffix": "/100", "font": {"size": 40, "color": decision_color}},
            delta={"reference": 50, "relative": True},
            domain={'x': [0, 0.5], 'y': [0.5, 1]}
        ), row=1, col=1)
        
        # 2. Sentiment pie
        fig.add_trace(go.Pie(
            labels=['Positive', 'Neutral', 'Negative'],
            values=[
                sentiment.get('positive', 0),
                sentiment.get('neutral', 0),
                sentiment.get('negative', 0)
            ],
            marker_colors=[
                self.colors['positive'],
                self.colors['neutral'],
                self.colors['negative']
            ],
            hole=0.4,
            textinfo='label+percent',
            hoverinfo='label+percent'
        ), row=1, col=2)
        
        # 3. Key metrics bar
        aspects = analysis.get('aspects', {}).get('details', {})
        metrics = {
            'Camera': aspects.get('camera', {}).get('sentiment', {}).get('positive', 50),
            'Battery': aspects.get('battery', {}).get('sentiment', {}).get('positive', 50),
            'Value': aspects.get('price', {}).get('sentiment', {}).get('positive', 50)
        }
        
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color=[
                self.colors['success'] if v >= 60 else self.colors['warning'] if v >= 40 else self.colors['danger']
                for v in metrics.values():
            ],
            text=[f'{v:.0f}%' for v in metrics.values()],
            textposition='auto'
        ), row=2, col=1)
        
        # 4. Confidence gauge
        confidence = analysis.get('confidence', 0.5) * 100
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={'text': "Analysis Confidence"},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=2, col=2)
        
        fig.update_layout(
            title="<b>Quick Decision Summary</b>",
            title_x=0.5,
            height=600,
            showlegend=False,
            paper_bgcolor='white',
            font=dict(family="Arial", size=12)
        )
        
        return fig
    
    def _determine_category(self, phone_model: str) -> str:
        """Determine phone category from model name"""
        model_lower = phone_model.lower()
        
        if any(word in model_lower for word in ['pro', 'max', 'ultra', 'plus']):
            return 'flagship'
        elif any(word in model_lower for word in ['lite', 'go', 'a', 'c', 'm']):
            return 'budget'
        else:
            return 'mid-range'
    
    def _get_confidence_label(self, confidence: float) -> str:
        """Get readable confidence label"""
        if confidence >= 0.75:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Moderate"
        elif confidence >= 0.2:
            return "Low"
        else:
            return "Very Low"


# Standalone function for easy integration
def create_decision_visualizations(:
    phone_model: str,
    analysis_result: Dict,
    confidence: float = 0.5
) -> Dict[str, go.Figure]:
    """
    Create all visualization charts for quick decision making
    
    Args:
        phone_model: Name of the phone
        analysis_result: Analysis results dictionary
        confidence: Confidence level (0-1)
    
    Returns:
        Dictionary of Plotly figures
    """
    visualizer = DecisionVisualizer()
    return visualizer.create_decision_dashboard(phone_model, analysis_result, confidence)


# Example usage
if __name__ == "__main__":
    # Sample analysis result
    sample_analysis = {
        'sentiment': {
            'positive': 72,
            'neutral': 18,
            'negative': 10
        },
        'aspects': {
            'details': {
                'camera': {
                    'sentiment': {'positive': 85, 'negative': 5, 'neutral': 10},
                    'confidence': 0.8,
                    'is_estimated': False
                },
                'battery': {
                    'sentiment': {'positive': 60, 'negative': 25, 'neutral': 15},
                    'confidence': 0.7,
                    'is_estimated': False
                },
                'performance': {
                    'sentiment': {'positive': 78, 'negative': 8, 'neutral': 14},
                    'confidence': 0.85,
                    'is_estimated': False
                },
                'price': {
                    'sentiment': {'positive': 45, 'negative': 40, 'neutral': 15},
                    'confidence': 0.6,
                    'is_estimated': True
                },
                'display': {
                    'sentiment': {'positive': 82, 'negative': 6, 'neutral': 12},
                    'confidence': 0.75,
                    'is_estimated': False
                }
            }
        },
        'data_quality': 'medium',
        'confidence': 0.75
    }
    
    # Create visualizations
    visualizer = DecisionVisualizer()
    figures = visualizer.create_decision_dashboard(
        "iPhone 15 Pro Max",
        sample_analysis,
        0.75
    )
    
    print(f"Created {len(figures)} visualization charts:")
    for name, fig in figures.items():
        print(f"  - {name}")
    
    # Show quick decision summary
    summary_fig = visualizer.create_quick_decision_summary(sample_analysis)
    # summary_fig.show()  # Uncomment to display in browser
