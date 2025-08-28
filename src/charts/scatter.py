import streamlit as st
import pandas as pd
import plotly.express as px
from helpers import helpers

class ScatterCharts:
    def __init__(self, connection):
        self.con = connection
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def scatter_chart_data_cpu_gpu_eff(
        _self,
        start_date,
        end_date,
        current_user,
        user_role,
        scale_efficiency=True,
        partition_selector=None,
        allowed_groups=None
    ):
        st.markdown('CPU & GPU Efficiency by Job Duration', help='Partition "jhub" and Interactive Jobs are excluded')
        
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # Only needed columns; prefix table with sqlite_db.
        query = """
            SELECT
                JobID,
                User,
                COALESCE(GpuUtil, 0) AS GpuUtil,
                (CPUEff * 100.0)      AS CPUEff,
                ROUND((CPUTime - TotalCPU) / 86400.0, 1) AS lost_cpu_days,
                Elapsed,
                Partition,
                Account
            FROM sqlite_db.allocations
            WHERE Start >= ?
              AND End   <= ?
              AND Partition != 'jhub'
              AND JobName  != 'interactive'
              AND Elapsed IS NOT NULL
        """
        params = [start_date, end_date]

        # Role-based filtering (admin shows longer jobs; uhh limited by account; user limited to own)
        if user_role == "admin":
            query += " AND Elapsed > 100"
        elif user_role == "uhh" and allowed_groups:
            query += " AND Account IN ({})".format(','.join('?' for _ in allowed_groups))
            params.extend(allowed_groups)
        elif user_role == "user":
            query += " AND User = ?"
            params.append(current_user)

        # Optional explicit user filter (keep if you sometimes pass current_user even for admins)
        if current_user and user_role != "user":
            query += " AND User = ?"
            params.append(current_user)

        if partition_selector:
            placeholders = ','.join(['?'] * len(partition_selector))
            query += f" AND Partition IN ({placeholders})"
            params.extend(partition_selector)

        query += " ORDER BY Elapsed ASC"

        # Execute via DuckDB
        df = _self.con.execute(query, params).fetchdf()

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return
        
        # Post-processing
        df['GpuUtil'] = df['GpuUtil'] * 100.0
        df['CPUEff'] = df['CPUEff'].clip(upper=100.0)

        if scale_efficiency:
            mask = df['CPUEff'] <= 100.0
            df.loc[mask, 'CPUEff'] = (df.loc[mask, 'CPUEff'] * 2.0).clip(upper=100.0)

        df['lost_cpu_days'] = df['lost_cpu_days'].fillna(0)

        # Keep a numeric Elapsed for x-axis scaling, but create a human-readable label for hover if you want
        df['Elapsed_human'] = df['Elapsed'].astype(int).apply(helpers.seconds_to_timestring)

        # Optional: concise custom hover
        hover_data = {
            'JobID': True,
            'User': True,
            'Partition': True,
            'lost_cpu_days': True,
            'CPUEff': True,
            'Elapsed_human': True
        }

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
            color='GpuUtil',
            range_color=[0, 100],
            color_continuous_scale=custom_color_scale,
            hover_data=hover_data,
            labels={'Elapsed': 'Elapsed Time (s)', 'CPUEff': 'CPU Efficiency (%)', 'Elapsed_human': 'Elapsed'}
        )
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(
            xaxis=dict(showgrid=False, title='Elapsed Time (s)'),
            yaxis=dict(showgrid=False, title='CPU Efficiency (%)'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, theme=None)
