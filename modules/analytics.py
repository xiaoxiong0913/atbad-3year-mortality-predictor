# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
  Module Name: Clinical Analytics Dashboard
  Description: 
      Visualizes aggregate statistics from the patient history database.
------------------------------------------------------------------------------
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class AnalyticsEngine:
    """
    Generates interactive charts for the Admin Dashboard.
    """
    
    def __init__(self, db_manager):
        self.db = db_manager

    def get_data(self):
        return self.db.fetch_all_records()

    def plot_risk_distribution(self):
        """
        Pie chart showing High Risk vs Low Risk ratio.
        """
        df = self.get_data()
        if df.empty: return None
        
        fig = px.pie(
            df, 
            names='risk_label', 
            title='Population Risk Stratification',
            color='risk_label',
            color_discrete_map={'High Risk':'#dc3545', 'Low Risk':'#28a745'},
            hole=0.4
        )
        fig.update_layout(height=350)
        return fig

    def plot_gender_stats(self):
        """
        Bar chart for Gender distribution.
        """
        df = self.get_data()
        if df.empty: return None
        
        fig = px.histogram(
            df, 
            x="gender", 
            color="risk_label",
            title="Risk Distribution by Gender",
            barmode='group',
            color_discrete_map={'High Risk':'#dc3545', 'Low Risk':'#28a745'}
        )
        fig.update_layout(height=350)
        return fig

    def plot_temporal_trend(self):
        """
        Line chart showing average risk score over time.
        """
        df = self.get_data()
        if df.empty: return None
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        daily_avg = df.groupby('date')['risk_prob'].mean().reset_index()
        
        fig = px.line(
            daily_avg, 
            x='date', 
            y='risk_prob',
            title='Daily Average Risk Trend',
            markers=True
        )
        fig.update_layout(height=350)
        return fig
