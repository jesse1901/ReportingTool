import streamlit as st
import pandas as pd
import plotly.express as px
from helpers import helpers

class ScatterCharts:
    def __init__(self, connection):
        self.con = connection
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def scatter_chart_data_cpu_gpu_eff(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None, allowed_groups=None):
        st.markdown('CPU & GPU Efficiency by Job Duration', help='Partition "jhub" and Interactive Jobs are excluded')
        
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # DuckDB Notes:
        # 1. "User", "Partition", "Account", "Start", "End", "Elapsed" must be quoted.
        # 2. Direct comparison for Unix Timestamps (Integers).
        # 3. Float division (86400.0).
        query = """
            SELECT "JobID", "User", COALESCE(GpuUtil, 0) AS GpuUtil, 
                (CPUEff * 100) AS CPUEff, 
                ROUND((CPUTime - TotalCPU) / 86400.0, 1) AS lost_cpu_days, 
                "Elapsed",
                "Partition"
            FROM allocations
            WHERE "Start" >= ? AND "End" <= ? 
            AND "Partition" != 'jhub'
            AND JobName != 'interactive'
            AND "Elapsed" IS NOT NULL
        """
        params = [start_date, end_date]

        # Role-based filtering with correct Quoting
        if user_role == "admin":
            query += ' AND "Elapsed" > 100'
        elif user_role == "uhh" and allowed_groups:
            placeholders = ','.join('?' for _ in allowed_groups)
            query += f' AND "Account" IN ({placeholders})'
            params.extend(allowed_groups)
        
        # User filter (handles role 'user' and specific user filters)
        if current_user:
            query += ' AND "User" = ?'
            params.append(current_user)

        if partition_selector:
            placeholders = ','.join(['?'] * len(partition_selector))
            query += f' AND "Partition" IN ({placeholders})'
            params.extend(partition_selector)

        # Add ORDER BY to the base query
        query += ' ORDER BY "Elapsed" ASC'
        
        try:
            df = _self.con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return
        
        # --- Pandas Post-Processing (same as original) ---
        
        # Vectorized operations for better performance
        df['GpuUtil'] = df['GpuUtil'] * 100
        df['Elapsed'] = df['Elapsed'].astype(int).apply(helpers.seconds_to_timestring)
        df['CPUEff'] = df['CPUEff'].clip(upper=100)

        if scale_efficiency:
            # More efficient conditional operation
            mask = df['CPUEff'] <= 100
            df.loc[mask, 'CPUEff'] = (df.loc[mask, 'CPUEff'] * 2).clip(upper=100)

        df['lost_cpu_days'] = df['lost_cpu_days'].fillna(0)

        # Vectorized hover text creation
        df['hover_text'] = (
            "JobID: " + df['JobID'].astype(str) + 
            "<br>User: " + df['User'].astype(str) + 
            "<br>Partition: " + df['Partition'].astype(str) + 
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
            hover_data={'JobID': True, 'User': True, 'Partition': True, 'lost_cpu_days': False, 'CPUEff': True},
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

