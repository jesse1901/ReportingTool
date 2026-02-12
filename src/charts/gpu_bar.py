import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers

class GpuBarCharts:
    def __init__(self, connection):
        self.con = connection
        
    @st.cache_data(ttl=600, show_spinner=False) 
    def bar_char_by_user(_self, start_date, end_date, current_user, user_role, number=None, partition_selector=None, allowed_groups=None) -> None:
        st.markdown('Total Lost GPU-Time per User', help='Partition "jhub" and Interactive Jobs are excluded. Only jobs with allocated GPUs are considered.')

        params = [start_date, end_date]

        base_conditions = """
            WHERE "Start" >= ? 
            AND "Start" IS NOT NULL
            AND "End" <= ? 
            AND "End" IS NOT NULL 
            AND "Partition" != 'jhub'
            AND JobName != 'interactive'
            AND NGPUS > 0
        """
        
        query = f"""
            SELECT 
                "User",
                COUNT(JobID) AS job_count,
                ROUND(SUM((NGPUS * Elapsed) * (1 - CASE WHEN GpuUtil BETWEEN 0 AND 1 THEN GpuUtil ELSE 0 END)) / 86400, 1) AS lost_gpu_days,
                STRING_AGG(DISTINCT "Account", ',') As Account
            FROM allocations
            {base_conditions}
        """

        if partition_selector:
            placeholders = ','.join(['?'] * len(partition_selector))
            query += f' AND "Partition" IN ({placeholders})'
            params.extend(partition_selector)

        if current_user:
            query += ' AND "User" = ?'
            params.append(current_user)

        if allowed_groups:
            placeholders = ','.join('?' for _ in allowed_groups)
            query += f' AND "Account" IN ({placeholders})'
            params.extend(allowed_groups)
        
        if number:
            query += f' GROUP BY "User" ORDER BY lost_gpu_days DESC LIMIT {number}'
        else:
            query += ' GROUP BY "User" ORDER BY lost_gpu_days DESC'

        try:
            result_df = _self.con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return
        
        if result_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        result_df['lost_gpu_days'] = result_df['lost_gpu_days'].clip(lower=0)
        
        max_lost_time = result_df['lost_gpu_days'].max()
        if max_lost_time > 0:
            tick_vals = np.linspace(0, max_lost_time, num=10)
            tick_text = [int(val) for val in tick_vals]
        else:
            tick_vals = [0]
            tick_text = [0]

        fig = px.bar(result_df, x='User', y='lost_gpu_days', hover_data=['job_count', 'Account'])

        fig.update_layout(
            xaxis=dict(
                title='User',
                tickangle=-45
            ),
            yaxis=dict(
                title='Total Lost GPU Time (in Days)',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickformat='d'
            )
        )

        st.plotly_chart(fig)
