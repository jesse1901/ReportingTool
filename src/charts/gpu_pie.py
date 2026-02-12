import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers
import duckdb

class GpuPieCharts:
    def __init__(self, db_path):
        self.db_path = db_path
        self.color_map = {
         'CANCELLED': "#803df5",
         'COMPLETED':  "#5ce488",
        'TIMEOUT': "#1c83e1",
        'FAILED': "#ff2b2b",
        'PREEMPTED': '#ffe312',
        'NODE_FAIL': '#566573'
        }
    
    @st.cache_data(ttl=600, show_spinner=False)
    def pie_chart_by_session_state(
        _self,
        start_date,
        end_date,
        current_user,
        user_role,
        partition_selector=None,
        allowed_groups=None
    ):
        query = f"""
            SELECT 
                CASE 
                    WHEN lower(State) LIKE 'cancelled%' THEN 'CANCELLED' 
                    ELSE State 
                END AS Category,
                SUM((NGPUS * Elapsed) * (1 - CASE WHEN GpuUtil BETWEEN 0 AND 1 THEN GpuUtil ELSE 0 END)) / 86400.0 AS lost_gpu_days
            FROM allocations
            WHERE "Partition" != 'jhub' 
            AND JobName != 'interactive'
            AND State NOT IN ('PENDING', 'RUNNING')
            AND "Start" >= ?
            AND "End"   <= ?
            AND "End" IS NOT NULL
            AND NGPUS > 0
        """
        params = [start_date, end_date]

        if partition_selector:
            placeholders = ','.join('?' for _ in partition_selector)
            query += f' AND "Partition" IN ({placeholders})'
            params.extend(partition_selector)
            
        if allowed_groups:
            placeholders = ','.join('?' for _ in allowed_groups)
            query += f' AND "Account" IN ({placeholders})'
            params.extend(allowed_groups)

        if current_user:
            query += ' AND "User" = ?'
            params.append(current_user)

        query += " GROUP BY Category"

        try:
            with duckdb.connect(_self.db_path, read_only=True) as con:
                df = con.execute(query, params).df()
        except Exception as e:
            st.error(f"Database Error: {e}")
            return

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        df['lost_gpu_days'] = df['lost_gpu_days'].clip(lower=0)
        df['lost_gpu_days'] = df['lost_gpu_days'].round()

        fig = px.pie(
            df,
            names="Category",
            values="lost_gpu_days",
            color="Category",
            color_discrete_map=_self.color_map,
        )
        st.markdown('Lost GPU Time by Job State', help='Partition "jhub" and Interactive Jobs are excluded. Only jobs with allocated GPUs are considered.')
        st.plotly_chart(fig)

        df_sorted = df.sort_values(by="lost_gpu_days", ascending=False)
        st.dataframe(df_sorted[["Category", "lost_gpu_days"]], hide_index=True, use_container_width=False)