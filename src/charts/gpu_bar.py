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
        st.markdown('Total GPU-Time per User', help='Partition "jhub" and Interactive Jobs are excluded. Only jobs with allocated GPUs are considered.')

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
                ROUND(SUM((NGPUS * Elapsed) * (1 - CASE WHEN GpuUtil BETWEEN 0 AND 1 THEN GpuUtil ELSE 0 END)) / 86400, 1) AS "Lost GPU Days",
                ROUND(SUM(NGPUS * Elapsed) / 86400, 1) AS "Total GPU Days",
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
            query += f' GROUP BY "User" ORDER BY "Total GPU Days" DESC LIMIT {number}'
        else:
            query += ' GROUP BY "User" ORDER BY "Total GPU Days" DESC'

        try:
            result_df = _self.con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return
        
        if result_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        

        result_df['Lost GPU Days'] = result_df['Lost GPU Days'].clip(lower=0)
        result_df['Total GPU Days'] = result_df['Total GPU Days'].clip(lower=0)
        result_df['Used GPU Days'] = result_df['Total GPU Days'] - result_df['Lost GPU Days']
        result_df['Used GPU Days'] = result_df['Used GPU Days'].clip(lower=0)


        fig = go.Figure()

        # 1. Add the "Used" bar (Bottom of the stack)
        fig.add_trace(go.Bar(
            name='Used GPU Days',
            x=result_df['User'],
            y=result_df['Used GPU Days'],
            marker_color='#5ce488',
            # Pass extra data for hover
            customdata=result_df[['job_count', 'Account', 'Total GPU Days']],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Used GPU Days: %{y}<br>" +
                "Job Count: %{customdata[0]}<br>" +
                "Account: %{customdata[1]}<br>" +
                "Total GPU Days: %{customdata[2]}" +
                "<extra></extra>" # Hides the secondary box
            )
        ))

        # 2. Add the "Lost" bar (Top of the stack)
        fig.add_trace(go.Bar(
            name='Lost GPU Days',
            x=result_df['User'],
            y=result_df['Lost GPU Days'],
            marker_color='#ff2b2b',
            customdata=result_df[['job_count', 'Account', 'Total GPU Days']],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Lost GPU Days: %{y}<br>" +
                "Job Count: %{customdata[0]}<br>" +
                "Account: %{customdata[1]}<br>" +
                "Total GPU Days: %{customdata[2]}" +
                "<extra></extra>"
            )
        ))

        # 3. Force the layout to stack
        fig.update_layout(
            barmode='stack',
            xaxis=dict(title='User', tickangle=-45),
            yaxis=dict(title='Total GPU Time (in Days)'),
            legend_title_text='Time Type',
            hovermode="x unified" # Optional: makes comparing easier
        )

        st.plotly_chart(fig)