import streamlit as st
import pandas as pd
import plotly.express as px
from helpers import helpers

class ScatterCharts:
    def __init__(self, connection):
        self.con = connection
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def scatter_chart_data_cpu_gpu_eff(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None):
        st.markdown('CPU & GPU Efficiency by Job Duration', help='Partition "jhub" and Interactive Jobs are excluded')
        
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # Optimized query - only select needed columns and add early filtering
        query = """
            SELECT JobID, User, COALESCE(GpuUtil, 0) AS GpuUtil, 
                (CPUEff * 100) AS CPUEff, 
                ROUND((CPUTime - TotalCPU) / 86400, 1) AS lost_cpu_days, 
                Elapsed
            FROM allocations
            WHERE Start >= ? AND End <= ? 
            AND Partition != 'jhub'
            AND JobName != 'interactive'
            AND Elapsed IS NOT NULL
        """
        params = [start_date, end_date]

        # Streamlined role-based filtering
        if user_role == "admin":
            query += " AND Elapsed > 100"
        elif user_role == "user":
            query += " AND User = ?"
            params.append(current_user)

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)

        # Add ORDER BY to the base query
        query += " ORDER BY Elapsed ASC"
        
        df = pd.read_sql_query(query, _self.con, params=params)

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return
        
        # Vectorized operations for better performance
        df['GpuUtil'] = df['GpuUtil'] * 100
        df['Elapsed'] = df['Elapsed'].astype(int).apply(helpers.seconds_to_timestring)
        df['CPUEff'] = df['CPUEff'].clip(upper=100)

        if scale_efficiency:
            # More efficient conditional operation
            mask = df['CPUEff'] <= 100
            df.loc[mask, 'CPUEff'] = (df.loc[mask, 'CPUEff'] * 2).clip(upper=100)

        df['lost_cpu_days'] = df['lost_cpu_days'].fillna(0)

        # Vectorized hover text creation - more efficient than apply
        df['hover_text'] = (
            "JobID: " + df['JobID'].astype(str) + 
            "<br>User: " + df['User'].astype(str) + 
            "<br>Lost CPU days: " + df['lost_cpu_days'].astype(str) + 
            "<br>CPU Efficiency: " + df['CPUEff'].astype(str)
        )

        # Pre-defined color scale to avoid recreation
        custom_color_scale = [
            [0.0, "CornflowerBlue"],  
            [0.25, "cyan"],
            [0.5, "green"],     
            [0.75, "orange"],  
            [1.0, "red"],       
        ]

        fig = px.scatter(
            df, 
            x='Elapsed', 
            y='CPUEff', 
            range_color=[0, 100],
            color='GpuUtil',
            color_continuous_scale=custom_color_scale,
            hover_data={'JobID': True, 'User': True, 'lost_cpu_days': True, 'CPUEff': True},
            labels={'Elapsed': 'Elapsed Time', 'CPUEff': 'CPU Efficiency (%)'}
        )
        
        # Batch update for better performance
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(
            xaxis=dict(showgrid=False, title='Elapsed Time'),
            yaxis=dict(showgrid=False, title='CPU Efficiency (%)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, theme=None)

