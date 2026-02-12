import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers
import plotly.graph_objects as go
import duckdb

class BarCharts:
    def __init__(self, db_path):
        self.db_path = db_path
        
    @st.cache_data(ttl=600, show_spinner=False) 
    def bar_char_by_user(_self, start_date, end_date, current_user, user_role, number=None, scale_efficiency=True, partition_selector=None, allowed_groups=None, use_log_scale=None) -> None:
        st.markdown('Total CPU-Time per User', help='Partition "jhub" and Interactive Jobs are excluded. Purple bars indicate lost CPU time on GPU nodes (excusable due to GPU workflow).')

        params = [start_date, end_date]

        base_conditions = """
            WHERE eff."Start" >= ? 
            AND eff."Start" IS NOT NULL
            AND eff."End" <= ? 
            AND eff."End" IS NOT NULL 
            AND slurm."Partition" != 'jhub'
            AND slurm.JobName != 'interactive'
        """
        
        # Logik für Scale Efficiency (nur für reine CPU Jobs ohne GPU)
        cpu_reserved_factor = 0.5 if scale_efficiency else 1.0
        
        # HIER: Nutzung von slurm.ngpus statt AllocTRES
        query = f"""
            SELECT 
                eff.User,
                COUNT(eff.JobID) AS job_count,
                
                ROUND(SUM(eff.cpu_s_used) / 86400, 1) AS "Used CPU Days",

                ROUND(SUM(
                    CASE 
                        WHEN slurm.ngpus IS NULL 
                        THEN ((eff.cpu_s_reserved * {cpu_reserved_factor}) - eff.cpu_s_used)
                        ELSE 0 
                    END
                ) / 86400, 1) AS "Lost CPU Days",

                ROUND(SUM(
                    CASE 
                        WHEN slurm.ngpus IS NOT NULL 
                        THEN (eff.cpu_s_reserved * {cpu_reserved_factor} - eff.cpu_s_used)
                        ELSE 0 
                    END
                ) / 86400, 1) AS "Lost GPU CPU Days",

                STRING_AGG(DISTINCT eff.Account, ',') As Account
            FROM eff
            JOIN slurm ON eff.JobID = slurm.JobID
            {base_conditions}
        """

        if partition_selector:
            placeholders = ','.join(['?'] * len(partition_selector))
            query += f' AND slurm."Partition" IN ({placeholders})'
            params.extend(partition_selector)

        if current_user:
            query += ' AND eff."User" = ?'
            params.append(current_user)

        if allowed_groups:
            placeholders = ','.join('?' for _ in allowed_groups)
            query += f' AND eff."Account" IN ({placeholders})'
            params.extend(allowed_groups)
        
        # Sortierlogik: Used + Lost (CPU) + Lost (GPU)
        order_logic = """
            "Used CPU Days" + 
            (CASE WHEN "Lost CPU Days" > 0 THEN "Lost CPU Days" ELSE 0 END) +
            (CASE WHEN "Lost GPU CPU Days" > 0 THEN "Lost GPU CPU Days" ELSE 0 END)
        """

        if number:
            query += f' GROUP BY eff."User" ORDER BY {order_logic} DESC LIMIT {number}'
        else:
            query += f' GROUP BY eff."User" ORDER BY {order_logic} DESC'

        try:
            with duckdb.connect(_self.db_path, read_only=True) as con:
                result_df = con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return
        
        if result_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Clipping negative values
        result_df['Lost CPU Days'] = result_df['Lost CPU Days'].clip(lower=0)
        result_df['Lost GPU CPU Days'] = result_df['Lost GPU CPU Days'].clip(lower=0)
        result_df['Used CPU Days'] = result_df['Used CPU Days'].clip(lower=0)
        
        # Total berechnen
        result_df['Total CPU Days'] = result_df['Lost CPU Days'] + result_df['Lost GPU CPU Days'] + result_df['Used CPU Days']
        
        fig = go.Figure()

        # Trace 1: Used (Grün)
        fig.add_trace(go.Bar(
            name='Used CPU Days',
            x=result_df['User'],
            y=result_df['Used CPU Days'],
            marker_color='#5ce488',
            customdata=result_df[['job_count', 'Account', 'Total CPU Days']],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Used CPU Days: %{y}<br>" +
                "Job Count: %{customdata[0]}<br>" +
                "Account: %{customdata[1]}<br>" +
                "Total CPU Days: %{customdata[2]}" +
                "<extra></extra>"
            )
        ))

        # Trace 2: Lost GPU (Lila)
        fig.add_trace(go.Bar(
            name='Lost CPU Days (GPU)',
            x=result_df['User'],
            y=result_df['Lost GPU CPU Days'],
            marker_color='#bf55ec',
            customdata=result_df[['job_count', 'Account', 'Total CPU Days']],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Lost CPU Days (GPU): %{y}<br>" +
                "<i>(Excusable due to GPU usage)</i><br>" +
                "Job Count: %{customdata[0]}<br>" +
                "Account: %{customdata[1]}<br>" +
                "Total CPU Days: %{customdata[2]}" +
                "<extra></extra>"
            )
        ))

        # Trace 3: Lost Standard (Rot)
        fig.add_trace(go.Bar(
            name='Lost CPU Days',
            x=result_df['User'],
            y=result_df['Lost CPU Days'],
            marker_color='#ff2b2b',
            customdata=result_df[['job_count', 'Account', 'Total CPU Days']],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Lost CPU Days: %{y}<br>" +
                "Job Count: %{customdata[0]}<br>" +
                "Account: %{customdata[1]}<br>" +
                "Total CPU Days: %{customdata[2]}" +
                "<extra></extra>"
            )
        ))

        y_axis_config = {'title': 'Total CPU Time (in Days)'}
        if use_log_scale:
            y_axis_config['type'] = 'log'

        fig.update_layout(
            barmode='stack',
            xaxis=dict(title='User', tickangle=-45),
            yaxis=y_axis_config,
            legend_title_text='Time Type',
            hovermode="x unified"
        )

        st.plotly_chart(fig)