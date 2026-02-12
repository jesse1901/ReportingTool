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


        df_melted = result_df.melt(id_vars=['User', 'job_count', 'Account', 'Total GPU Days'], 
                                   value_vars=['Used GPU Days', 'Lost GPU Days'],
                                   var_name='Time Type', 
                                   value_name='GPU Days')

        # Create the figure WITHOUT barmode here
        fig = px.bar(df_melted, 
                     x='User', 
                     y='GPU Days', 
                     color='Time Type',
                     hover_data=['job_count', 'Account', 'Total GPU Days'],
                     color_discrete_map={'Used GPU Days': '#5ce488', 'Lost GPU Days': '#ff2b2b'})

        # Force stacking in the layout
        fig.update_layout(
            barmode='stack',  # <--- This is the mandatory switch
            xaxis=dict(title='User', tickangle=-45),
            yaxis=dict(title='Total GPU Time (in Days)')
        )

        st.plotly_chart(fig)