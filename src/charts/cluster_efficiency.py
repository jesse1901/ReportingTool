import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go

class ClusterEfficiencyCharts:
    def __init__(self, db_path):
        self.db_path = db_path

    def display_cluster_efficiency(self, start_date, end_date, scale_efficiency):
        st.header("Cluster Efficiency")
        col1, col2 = st.columns([1,1])
        # CPU Chart
        with col1:
            self.display_cpu_efficiency_chart(start_date, end_date, scale_efficiency)

        # GPU Chart
        with col2: 
            self.display_gpu_efficiency_chart(start_date, end_date)

    @st.cache_data(ttl=600, show_spinner=False)
    def get_cpu_efficiency_data(_self, start_date, end_date, scale_efficiency):
        
        cpu_query = f"""
        WITH cpu_data AS (
            SELECT 
                SUM(CASE WHEN slurm. => 0 THEN eff.cpu_s_used ELSE 0 END) AS cpu_with_gpu_seconds,
                SUM(CASE WHEN slurm.NGPUS = 0 THEN eff.cpu_s_used ELSE 0 END) AS used_cpu_seconds,
                SUM(eff.cpu_s_reserved - eff.cpu_s_used) AS lost_cpu_seconds
            FROM eff
            JOIN slurm ON eff.JobID = slurm.JobID
            WHERE eff."Start" >= ? AND eff."End" <= ?
            AND eff."End" IS NOT NULL
            AND slurm."Partition" != 'jhub'
            AND slurm.JobName != 'interactive'
        )
        SELECT
            (cpu_with_gpu_seconds / 86400.0) * {0.5 if scale_efficiency else 1.0} AS cpu_with_gpu_days,
            (used_cpu_seconds / 86400.0) * {0.5 if scale_efficiency else 1.0} AS used_cpu_days,
            (lost_cpu_seconds / 86400.0) * {0.5 if scale_efficiency else 1.0} AS lost_cpu_days
        FROM cpu_data;
        """
        
        with duckdb.connect(_self.db_path, read_only=True) as con:
            cpu_df = con.execute(cpu_query, [start_date, end_date]).df()

        return cpu_df

    def display_cpu_efficiency_chart(self, start_date, end_date, scale_efficiency):
        cpu_df = self.get_cpu_efficiency_data(start_date, end_date, scale_efficiency)

        if cpu_df.empty:
            st.warning("No CPU data available for the selected time range.")
            return

        cpu_with_gpu_days = cpu_df['cpu_with_gpu_days'].iloc[0]
        used_cpu_days = cpu_df['used_cpu_days'].iloc[0]
        lost_cpu_days = cpu_df['lost_cpu_days'].iloc[0]
        
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='CPU with GPU',
            x=['Cluster'],
            y=[cpu_with_gpu_days],
            marker_color='#5ce488'
        ))
        
        fig.add_trace(go.Bar(
            name='Used CPU',
            x=['Cluster'],
            y=[used_cpu_days],
            marker_color='#1c83e1'
        ))

        fig.add_trace(go.Bar(
            name='Lost CPU',
            x=['Cluster'],
            y=[lost_cpu_days],
            marker_color='#ff2b2b'
        ))
        
        fig.update_layout(
            barmode='stack',
            title_text='CPU Usage',
            yaxis_title='CPU Days',
            xaxis_title='Cluster',
            legend_title_text='Usage Type'
        )

        st.plotly_chart(fig)


    @st.cache_data(ttl=600, show_spinner=False)
    def get_gpu_efficiency_data(_self, start_date, end_date):
        gpu_query = """
        SELECT 
            SUM(NGPUS * Elapsed * GpuUtil) / 86400.0 AS used_gpu_days,
            SUM(NGPUS * Elapsed * (1 - GpuUtil)) / 86400.0 AS unused_gpu_days
        FROM allocations
        WHERE "Start" >= ? AND "End" <= ?
        AND "End" IS NOT NULL
        AND NGPUS > 0
        AND "Partition" != 'jhub'
        AND JobName != 'interactive'
        """

        with duckdb.connect(_self.db_path, read_only=True) as con:
            gpu_df = con.execute(gpu_query, [start_date, end_date]).df()

        return gpu_df

    def display_gpu_efficiency_chart(self, start_date, end_date):
        gpu_df = self.get_gpu_efficiency_data(start_date, end_date)

        if gpu_df.empty:
            st.warning("No GPU data available for the selected time range.")
            return

        used_gpu_days = gpu_df['used_gpu_days'].iloc[0]
        unused_gpu_days = gpu_df['unused_gpu_days'].iloc[0]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Used GPU',
            x=['Cluster'],
            y=[used_gpu_days],
            marker_color='#5ce488'
        ))

        fig.add_trace(go.Bar(
            name='Unused GPU',
            x=['Cluster'],
            y=[unused_gpu_days],
            marker_color='#ff2b2b'
        ))

        fig.update_layout(
            barmode='stack',
            title_text='GPU Usage',
            yaxis_title='GPU Days',
            xaxis_title='Cluster',
            legend_title_text='Usage Type'
        )

        st.plotly_chart(fig)

