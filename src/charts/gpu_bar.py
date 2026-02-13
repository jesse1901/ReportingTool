import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers
import plotly.graph_objects as go
import duckdb

class GpuBarCharts:
    def __init__(self, db_path):
        self.db_path = db_path
        
    @st.cache_data(ttl=600, show_spinner=False) 
    def bar_chart_by_user_gpu(_self, start_date, end_date, current_user, user_role, number=None, partition_selector=None, allowed_groups=None, scale_type="Absolute", sort_by="Total GPU", sort_by_percentage=False, min_total_days=0) -> None:
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
                
                ROUND(SUM(GREATEST((NGPUS * Elapsed) * (CASE WHEN GpuUtil BETWEEN 0 AND 1 THEN GpuUtil ELSE 0 END), 0)) / 86400, 1) AS "Used GPU Days",
                
                ROUND(SUM(GREATEST((NGPUS * Elapsed) * (1 - CASE WHEN GpuUtil BETWEEN 0 AND 1 THEN GpuUtil ELSE 0 END), 0)) / 86400, 1) AS "Lost GPU Days",
                
                ROUND(SUM(GREATEST(NGPUS * Elapsed, 0)) / 86400, 1) AS "Total GPU Days",
                
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
        
        # Dynamische Sortierlogik (Absolut vs. Prozentual)
        total_gpu_expr = '("Used GPU Days" + "Lost GPU Days")'
        
        if sort_by_percentage:
            sort_mapping = {
                "Used GPU": f'"Used GPU Days" / NULLIF({total_gpu_expr}, 0)',
                "Lost GPU": f'"Lost GPU Days" / NULLIF({total_gpu_expr}, 0)',
                "Total GPU": total_gpu_expr 
            }
        else:
            sort_mapping = {
                "Used GPU": '"Used GPU Days"',
                "Lost GPU": '"Lost GPU Days"',
                "Total GPU": total_gpu_expr
            }
            
        order_col = sort_mapping.get(sort_by, total_gpu_expr)

        # Zusammenbau von GROUP BY, HAVING (Filter) und ORDER BY
        query += ' GROUP BY "User"'

        # Filter für User mit weniger als X GPU Days (HAVING)
        if min_total_days > 0:
            query += ' HAVING ROUND(SUM(GREATEST(NGPUS * Elapsed, 0)) / 86400, 1) >= ?'
            params.append(min_total_days)

        query += f' ORDER BY {order_col} DESC'

        if number:
            query += f' LIMIT {number}'

        # Datenbank-Ausführung
        try:
            with duckdb.connect(_self.db_path, read_only=True) as con:
                result_df = con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return
        
        if result_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Clipping negative values zur Sicherheit
        result_df['Lost GPU Days'] = result_df['Lost GPU Days'].clip(lower=0)
        result_df['Used GPU Days'] = result_df['Used GPU Days'].clip(lower=0)
        
        # Total für den Hover dynamisch neu berechnen
        result_df['Total GPU Days'] = result_df['Used GPU Days'] + result_df['Lost GPU Days']

        fig = go.Figure()

        # 1. Add the "Used" bar (Bottom of the stack)
        fig.add_trace(go.Bar(
            name='Used GPU Days',
            x=result_df['User'],
            y=result_df['Used GPU Days'],
            marker_color='#5ce488',
            customdata=result_df[['job_count', 'Account', 'Total GPU Days']],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Used GPU Days: %{y}<br>" +
                "Job Count: %{customdata[0]}<br>" +
                "Account: %{customdata[1]}<br>" +
                "Total GPU Days: %{customdata[2]}" +
                "<extra></extra>" 
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

        # -------------------------------------------------------------------
        # Layout-Konfiguration
        # -------------------------------------------------------------------
        barmode_selection = 'group' if bar_mode == 'Grouped' else 'stack'
        
        y_axis_config = {'title': 'Total GPU Time (in Days)'}
        barnorm_setting = None

        if scale_type == 'Log':
            y_axis_config['type'] = 'log'
        elif scale_type == 'Percentage':
            barnorm_setting = 'percent'
            y_axis_config['title'] = 'Percentage of GPU Time'
            y_axis_config['ticksuffix'] = '%'
            barmode_selection = 'stack'

        fig.update_layout(
            barmode=barmode_selection,
            barnorm=barnorm_setting,
            bargap=0,
            xaxis=dict(title='User', tickangle=-45),
            yaxis=y_axis_config,
            legend_title_text='Time Type',
            hovermode="x unified" 
        )

        st.plotly_chart(fig)