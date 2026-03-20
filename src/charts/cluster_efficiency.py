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
        # Scale Efficiency Faktor: 0.5 wenn aktiv, sonst 1.0
        # Dieser Faktor gilt NUR für "Lost CPU Time" auf Nicht-GPU Nodes.
        cpu_scale_factor = 0.5 if scale_efficiency else 1.0
        
        cpu_query = f"""
        SELECT 
            -- 1. Total Used CPU Days (Grün) - Keine Skalierung
            SUM(eff.cpu_s_used) / 86400.0 AS used_cpu_days,

            -- 2. Lost CPU Days auf CPU Nodes (Rot) - Hier greift scale_efficiency
            -- Bedingung: GpuUtil IS NULL (bedeutet keine GPU)
            SUM(
                CASE 
                    WHEN slurm.GpuUtil IS NULL 
                    THEN (eff.cpu_s_reserved * {cpu_scale_factor} )  - eff.cpu_s_used 
                    ELSE 0 
                END
            ) / 86400.0 AS lost_cpu_days,

            -- 3. Lost CPU Days auf GPU Nodes (Lila) - Hier KEINE Skalierung
            -- Bedingung: GpuUtil IS NOT NULL (bedeutet GPU Job)
            SUM(
                CASE 
                    WHEN slurm.GpuUtil IS NOT NULL 
                    THEN (eff.cpu_s_reserved - eff.cpu_s_used)
                    ELSE 0 
                END
            ) / 86400.0 AS lost_gpu_node_cpu_days

        FROM eff
        JOIN slurm ON eff.JobID = slurm.JobID
        WHERE eff."Start" >= ? AND eff."End" <= ?
        AND eff."End" IS NOT NULL
        AND slurm."Partition" != 'jhub'
        AND slurm.JobName != 'interactive'
        """
        
        with duckdb.connect(_self.db_path, read_only=True) as con:
            cpu_df = con.execute(cpu_query, [start_date, end_date]).df()

        return cpu_df

    def display_cpu_efficiency_chart(self, start_date, end_date, scale_efficiency):
        cpu_df = self.get_cpu_efficiency_data(start_date, end_date, scale_efficiency)

        if cpu_df.empty:
            st.warning("No CPU data available for the selected time range.")
            return

        # Werte extrahieren (mit Fallback auf 0 falls None)
        used_cpu_days = cpu_df['used_cpu_days'].iloc[0] or 0
        lost_cpu_days = cpu_df['lost_cpu_days'].iloc[0] or 0
        lost_gpu_node_cpu_days = cpu_df['lost_gpu_node_cpu_days'].iloc[0] or 0
        
        # Negative Werte bereinigen (Clipping)
        used_cpu_days = max(0, used_cpu_days)
        lost_cpu_days = max(0, lost_cpu_days)
        lost_gpu_node_cpu_days = max(0, lost_gpu_node_cpu_days)

        fig = go.Figure()
        
        # 1. Used CPU (Grün)
        fig.add_trace(go.Bar(
            name='Used CPU',
            x=['Cluster'],
            y=[used_cpu_days],
            marker_color='#5ce488',
            hovertemplate="Used CPU Days: %{y:.1f}<extra></extra>"
        ))

        # 2. Lost CPU auf GPU Nodes (Lila) - Entschuldbar
        fig.add_trace(go.Bar(
            name='Lost CPU (GPU Context)',
            x=['Cluster'],
            y=[lost_gpu_node_cpu_days],
            marker_color='#bf55ec',
            hovertemplate="Lost CPU (GPU Context): %{y:.1f}<br><i>(Excusable due to GPU usage)</i><extra></extra>"
        ))

        # 3. Lost CPU Standard (Rot)
        fig.add_trace(go.Bar(
            name='Lost CPU',
            x=['Cluster'],
            y=[lost_cpu_days],
            marker_color='#ff2b2b',
            hovertemplate="Lost CPU Days: %{y:.1f}<extra></extra>"
        ))
        
        fig.update_layout(
            barmode='stack',
            title_text='Overall CPU Usage',
            yaxis_title='CPU Days',
            xaxis_title='',
            legend_title_text='Usage Type'
        )

        st.plotly_chart(fig)


    @st.cache_data(ttl=600, show_spinner=False)
    def get_gpu_efficiency_data(_self, start_date, end_date):
        # Hier wird GpuUtil IS NOT NULL verwendet, um sicherzustellen, dass nur Jobs mit GPUs betrachtet werden.
        # Scale Efficiency wird hier NICHT angewendet.
        gpu_query = """
        SELECT 
            SUM(NGPUS * Elapsed * GpuUtil) / 86400.0 AS used_gpu_days,
            SUM(NGPUS * Elapsed * (1 - GpuUtil)) / 86400.0 AS unused_gpu_days
        FROM allocations
        WHERE "Start" >= ? AND "End" <= ?
        AND "End" IS NOT NULL
        AND GpuUtil IS NOT NULL
        AND "Partition" != 'jhub'
        AND JobName != 'interactive'
        """

        with duckdb.connect(_self.db_path, read_only=True) as con:
            gpu_df = con.execute(gpu_query, [start_date, end_date]).df()

        return gpu_df

    def display_gpu_efficiency_chart(self, start_date, end_date):
        gpu_df = self.get_gpu_efficiency_data(start_date, end_date)

        if gpu_df.empty or gpu_df['used_gpu_days'].iloc[0] is None:
            st.warning("No GPU data available for the selected time range.")
            return

        used_gpu_days = gpu_df['used_gpu_days'].iloc[0] or 0
        unused_gpu_days = gpu_df['unused_gpu_days'].iloc[0] or 0

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
            title_text='Overall GPU Usage',
            yaxis_title='GPU Days',
            xaxis_title='',
            legend_title_text='Usage Type'
        )

        st.plotly_chart(fig)