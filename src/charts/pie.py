import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers

class PieCharts:
    def __init__(self, connection):
        self.con = connection
        self.color_map = {
         'CANCELLED': "#803df5", #'#1f77b4 ',    # Light Blue
         'COMPLETED':  "#5ce488",  ##17becf ',    # Light Sky Blue
        'TIMEOUT': "#1c83e1", #'#d62728 ',     # red
        'FAILED': "#ff2b2b",      # Pink
        'PREEMPTED': '#ffe312',     # Light Green
        'NODE_FAIL': '#566573'
    }
        
    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_session_state(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None):
        # Common base query parts
        base_conditions = """
            WHERE Partition != 'jhub' 
            AND State NOT IN ('PENDING', 'RUNNING') 
            AND Start >= ? 
            AND End <= ? 
            AND JobName != 'interactive'
        """
        
        if scale_efficiency:
            query = f"""
                SELECT 
                    IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
                    CAST(ROUND(SUM((CPUTime * 0.5) - TotalCPU) / 86400, 1) AS INTEGER) AS lost_cpu_days
                FROM allocations
                {base_conditions}
            """
        else:
            query = f"""
                SELECT 
                    IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
                    CAST(ROUND(SUM((CPUTime - TotalCPU) / 86400)) AS INTEGER) AS lost_cpu_days
                FROM allocations
                {base_conditions}
            """
        
        params = [start_date, end_date]

        # Streamlined role filtering
        if user_role == "user":
            query += " AND User = ?"
            params.append(current_user)

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)

        # Add GROUP BY to the base query
        query += " GROUP BY Category"
        
        df = pd.read_sql_query(query, _self.con, params=params)
        df['lost_cpu_days'] = df['lost_cpu_days'].clip(lower=0)
        
        fig = px.pie(
            df,
            names='Category',
            values='lost_cpu_days',
            color='Category',
            color_discrete_map=_self.color_map,
        )
        
        st.markdown('Lost CPU Time by Job State', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)
        
        # Since df is already grouped by Category, we can directly sort without regrouping
        df_sorted = df.sort_values(by='lost_cpu_days', ascending=False)
        st.dataframe(df_sorted, hide_index=True, use_container_width=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_job_count(_self, start_date, end_date, current_user, user_role, partition_selector=None):
        query = """
            SELECT
                IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category, 
                COUNT(JobID) AS JobCount
            FROM allocations
            WHERE Partition != 'jhub' 
            AND State NOT IN ('PENDING', 'RUNNING') 
            AND Start >= ? 
            AND End <= ? 
            AND JobName != 'interactive'
        """
        params = [start_date, end_date]

        # Streamlined role filtering
        if user_role == 'user':
            query += " AND User = ?"
            params.append(current_user)

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)
        
        # GROUP BY the Category (not State) to properly aggregate CANCELLED entries
        query += " GROUP BY IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State)"
        
        df = pd.read_sql_query(query, _self.con, params=params)

        fig = px.pie(
            df,
            names='Category',
            values='JobCount',
            color='Category',
            color_discrete_map=_self.color_map
        )
        fig.update_layout(showlegend=False)
        
        st.markdown('Job Count by Job State', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)
        
        # Since df is already properly grouped by Category, we can directly sort
        df_sorted = df.sort_values(by='JobCount', ascending=False)
        st.dataframe(df_sorted, hide_index=True, use_container_width=False)

 
    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_job_runtime(_self, start_date, end_date, scale_efficiency=True, partition_selector=None) -> None:
        # Common WHERE conditions to avoid duplication
        base_conditions = "WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'"
        params = [start_date, end_date]
        
        if partition_selector:
            base_conditions += " AND Partition = ?"
            params.append(partition_selector)

        if scale_efficiency:
            # Combined query to get both pie chart data and summary stats in one call
            combined_query = f"""
                SELECT
                    (End - Start) / 60 AS runtime_minutes,
                    ROUND(((CPUTime * 0.5) - TotalCPU) / 86400, 1) AS lost_cpu_days,
                    CPUTime,
                    TotalCPU
                FROM allocations
                {base_conditions}
            """
        else:
            combined_query = f"""
                SELECT
                    (End - Start) / 60 AS runtime_minutes,
                    ROUND((CPUTime - TotalCPU) / 86400, 1) AS lost_cpu_days,
                    CPUTime,
                    TotalCPU
                FROM allocations
                {base_conditions}
            """

        df = pd.read_sql_query(combined_query, _self.con, params=params)
        
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        # Clip negative values early
        df['lost_cpu_days'] = df['lost_cpu_days'].clip(lower=0)
        
        # Pre-calculate summary statistics from the main dataset
        if scale_efficiency:
            total_cpu_days = int(round(df['CPUTime'].sum() / 86400 * 0.5))
            lost_cpu_days_total = max(0, int(round(((df['CPUTime'] * 0.5) - df['TotalCPU']).sum() / 86400)))
        else:
            total_cpu_days = int(round(df['CPUTime'].sum() / 86400))
            lost_cpu_days_total = max(0, int(round((df['CPUTime'] - df['TotalCPU']).sum() / 86400)))
        
        # Calculate efficiency
        cluster_efficiency = (total_cpu_days - lost_cpu_days_total) / total_cpu_days * 100 if total_cpu_days > 0 else 0
        
        # Optimize bin creation
        max_runtime = df['runtime_minutes'].max()
        predefined_bins = [0, 2, 5, 10, 20, 60, 120, 240, 480, 1440, 2880, 5760, 11520, 23040]
        bins = [b for b in predefined_bins if b <= max_runtime] + [max_runtime]
        bins = sorted(set(bins))  # Remove duplicates and sort

        # Create intervals and group efficiently
        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins)
        cpu_time_by_interval = df.groupby('runtime_interval', observed=True)['lost_cpu_days'].sum().reset_index()
        cpu_time_by_interval['runtime_interval'] = cpu_time_by_interval['runtime_interval'].apply(helpers.format_interval_label)

        # Create pie chart
        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='lost_cpu_days')
        
        st.markdown('Lost CPU Time by Job Runtime Interval', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)

        # Create summary dataframe efficiently without second query
        summary_data = {
            'total CPU days booked': f"{total_cpu_days:,}",
            'total CPU days lost': f"{lost_cpu_days_total:,}",
            'cluster efficiency': f"{cluster_efficiency:.2f}%"
        }
        
        df2 = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Time in days'])
        df2 = df2.set_index('Metric')
        
        st.write('Cluster Efficiency')
        st.dataframe(df2, use_container_width=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_batch_inter(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None) -> None:
        # Build query with optimized conditions
        if scale_efficiency:
            query = """
                SELECT
                    ROUND(((CPUTime * 0.5) - TotalCPU) / 86400, 1) AS lost_cpu_days,
                    JobName
                FROM allocations 
                WHERE Partition != 'jhub' 
                AND Start >= ? 
                AND End <= ?
                AND JobName != 'interactive'
            """
        else:
            query = """
                SELECT             
                    ROUND((CPUTime - TotalCPU) / 86400) AS lost_cpu_days,
                    JobName
                FROM allocations 
                WHERE Partition != 'jhub' 
                AND Start >= ? 
                AND End <= ? 
                AND JobName != 'interactive'
            """
        
        params = [start_date, end_date]

        # Add filters efficiently
        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)
        
        if user_role == 'user':
            query += " AND User = ?"
            params.append(current_user)

        df = pd.read_sql_query(query, _self.con, params=params)
        
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        # Optimized category assignment using vectorized operations
        conditions = [
            df['JobName'] == 'spawner-jupyterhub',
            df['JobName'] == 'interactive', 
            df['JobName'] != ''
        ]
        choices = ['Jupyterhub', 'Interactive', 'Batch']
        df['Category'] = np.select(conditions, choices, default='None')

        # Group and aggregate efficiently
        aggregated_df = df.groupby('Category', as_index=False)['lost_cpu_days'].sum()
        
        # Pre-defined color mapping
        color_map = {
            'Interactive': 'red',
            'Batch': 'darkcyan',
            'None': 'grey',
            'Jupyterhub': 'orange'
        }

        # Create pie chart
        fig = px.pie(
            aggregated_df,
            names='Category',
            values='lost_cpu_days',
            color='Category',
            color_discrete_map=color_map
        )
        
        st.markdown('Lost CPU Time by Job Category', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)
        st.dataframe(aggregated_df, hide_index=True, use_container_width=False)
