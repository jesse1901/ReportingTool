import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers

class BarCharts:
    def __init__(self, connection):
        self.con = connection

    @st.cache_data(ttl=3600, show_spinner=False)
    def bar_char_by_user(_self, start_date, end_date, current_user, user_role, number=None, scale_efficiency=True, partition_selector=None, allowed_groups=None) -> None:
        st.markdown('Total Lost CPU-Time per User', help='Partition "jhub" and Interactive Jobs are excluded')

        # Set parameters for the SQL query
        params = [start_date, end_date]

        # Build query with common filters first
        # DuckDB: Quote reserved keywords "Start", "End", "Partition"
        base_conditions = """
            WHERE eff."Start" >= ? 
            AND eff."Start" IS NOT NULL
            AND eff."End" <= ? 
            AND eff."End" IS NOT NULL 
            AND slurm."Partition" != 'jhub'
            AND slurm.JobName != 'interactive'
        """
        
        # DuckDB: 
        # 1. Quote "User", "Account"
        # 2. Use ANY_VALUE(eff."Account") because we are grouping by User only. 
        #    SQLite allows selecting non-grouped columns loosely; DuckDB is strict.
        if scale_efficiency:
            query = f"""
                SELECT 
                    eff.User,
                    COUNT(eff.JobID) AS job_count,
                    ROUND(SUM((eff.cpu_s_reserved * 0.5) - eff.cpu_s_used) / 86400, 1) AS lost_cpu_days,
                    ANY_VALUE(eff.Account) AS Account
                FROM eff
                JOIN slurm ON eff.JobID = slurm.JobID
                {base_conditions}
            """
        else:
            query = f"""
                SELECT 
                    eff."User",
                    ROUND(SUM((eff.cpu_s_reserved - eff.cpu_s_used) / 86400), 1) AS lost_cpu_days,
                    COUNT(eff.JobID) AS job_count,
                    ROUND(SUM(eff.Elapsed * eff.NCPUS) / 86400, 1) AS total_cpu_days,
                    ANY_VALUE(eff."Account") AS "Account"
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
        
        # Add grouping and ordering with LIMIT in SQL if number is specified
        if number:
            query += f' GROUP BY eff."User" ORDER BY lost_cpu_days DESC LIMIT {number}'
        else:
            query += ' GROUP BY eff."User" ORDER BY lost_cpu_days DESC'

        # Execute query using native DuckDB method for speed
        try:
            result_df = _self.con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return
        
        # Early exit if no data
        if result_df.empty:
            st.warning("No data available for the selected criteria.")
            return
        
        # Clip negative values
        result_df['lost_cpu_days'] = result_df['lost_cpu_days'].clip(lower=0)
        
        # Calculate ticks more efficiently
        max_lost_time = result_df['lost_cpu_days'].max()
        if max_lost_time > 0:
            tick_vals = np.linspace(0, max_lost_time, num=10)
            tick_text = [int(val) for val in tick_vals]
        else:
            tick_vals = [0]
            tick_text = [0]

        # Create chart with optimized settings
        fig = px.bar(result_df, x='User', y='lost_cpu_days', hover_data=['job_count', 'Account'])

        fig.update_layout(
            xaxis=dict(
                title='User',
                tickangle=-45
            ),
            yaxis=dict(
                title='Total Lost CPU Time (in Days)',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                tickformat='d'
            )
        )

        st.plotly_chart(fig)
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def job_counts_by_log2(_self, start_date, end_date, number, partition_selector, user_role, current_user=None, allowed_groups=None) -> None:
        st.markdown('Job Count by Job Time', help='Partition "jhub" and Interactive Jobs are excluded')

        min_runtime = max(0, number)  # minutes

        query = """
            SELECT ("End" - "Start") / 60.0 AS runtime_minutes
            FROM allocations
            WHERE "Partition" != 'jhub'
            AND JobName != 'interactive'
            AND ("End" - "Start") / 60.0 >= ?
            AND "Start" >= ?
            AND "End"   <= ?
            AND "End" IS NOT NULL
        """
        params = [min_runtime, start_date, end_date]


        query, params = helpers.build_conditions(query, params, partition_selector, allowed_groups)

        if current_user:
            query += ' AND "User" = ?'
            params.append(current_user)

        try:
            df = _self.con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        # Data type safety
        df['runtime_minutes'] = pd.to_numeric(df['runtime_minutes'], errors='coerce').fillna(0)

        max_runtime = float(df['runtime_minutes'].max())

        # Predefined Bins in MINUTES
        all_bins = [0, 2, 5, 10, 20, 60, 120, 240, 480, 1440, 2880, 5760, 11520, 23040]
        edges = [b for b in all_bins if b >= min_runtime and b < max_runtime]
        if not edges or edges[0] > min_runtime:
            edges.insert(0, min_runtime)

        # Last edge slightly above max_runtime to include max values
        EPS = 1e-9
        edges.append(max_runtime + EPS)

        df['runtime_interval'] = pd.cut(
            df['runtime_minutes'],
            bins=edges,
            include_lowest=True,
            right=False
        )

        # Format Labels
        df['runtime_interval'] = df['runtime_interval'].apply(helpers.format_interval_label)

        # Counting
        job_counts = df['runtime_interval'].value_counts().sort_index()
        job_counts_df = job_counts.reset_index()
        job_counts_df.columns = ['runtime_interval', 'job_count']

        # Plot
        fig = px.bar(
            job_counts_df,
            x="runtime_interval",
            y="job_count",
            width=550,
            height=450,
        )
        fig.update_layout(
            xaxis_tickangle=-90,
            yaxis_title=None,
            xaxis=dict(title=None, tickfont=dict(size=15)),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=False)
