import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from helpers import helpers

class PieCharts:
    def __init__(self, connection):
        self.con = connection
        self.color_map = {
            'CANCELLED': "#803df5",
            'COMPLETED': "#5ce488",
            'TIMEOUT':   "#1c83e1",
            'FAILED':    "#ff2b2b",
            'PREEMPTED': '#ffe312',
            'NODE_FAIL': '#566573'
        }
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_session_state(
        _self,
        start_date,
        end_date,
        current_user,
        user_role,
        scale_efficiency=True,
        partition_selector=None,
        allowed_groups=None
    ):
        scale = "0.5" if scale_efficiency else "1.0"
        query = f"""
            SELECT 
                CASE WHEN LOWER(State) LIKE 'cancelled %%' THEN 'CANCELLED' ELSE State END AS Category,
                ((CPUTime * {scale}) - TotalCPU) / 86400.0 AS lost_cpu_days 
            FROM sqlite_db.allocations
            WHERE Partition != 'jhub' 
              AND JobName != 'interactive'
              AND State NOT IN ('PENDING', 'RUNNING')
              AND Start >= ? 
              AND End   <= ?
              AND End IS NOT NULL
        """
        params = [start_date, end_date]

        query, params = helpers.build_conditions(
            query, params, partition_selector, allowed_groups, user_role, current_user
        )

        if current_user:
            query += " AND User = ?"
            params.append(current_user)

        df = _self.con.execute(query, params).fetchdf()
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        grouped_raw = df.groupby("Category", as_index=False)["lost_cpu_days"].sum()
        grouped_raw.rename(columns={"lost_cpu_days": "lost_days_raw"}, inplace=True)

        net_total = grouped_raw["lost_days_raw"].sum()

        if net_total <= 0:
            grouped_raw["lost_cpu_days"] = 0.0
        else:
            pos_sum = grouped_raw.loc[grouped_raw["lost_days_raw"] > 0, "lost_days_raw"].sum()
            if pos_sum > 0:
                scale_factor = net_total / pos_sum
                grouped_raw["lost_cpu_days"] = grouped_raw["lost_days_raw"].clip(lower=0.0) * scale_factor
            else:
                grouped_raw["lost_cpu_days"] = 0.0

        grouped_raw["lost_cpu_days"] = grouped_raw["lost_cpu_days"].round(1)

        fig = px.pie(
            grouped_raw,
            names="Category",
            values="lost_cpu_days",
            color="Category",
            color_discrete_map=_self.color_map,
        )
        st.markdown('Lost CPU Time by Job State', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)

        df_sorted = grouped_raw.sort_values(by="lost_cpu_days", ascending=False)
        st.dataframe(df_sorted[["Category", "lost_cpu_days"]], hide_index=True, use_container_width=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_job_count(
        _self, start_date, end_date, current_user, user_role, partition_selector=None, allowed_groups=None
    ):
        query = """
            SELECT
                CASE WHEN LOWER(State) LIKE 'cancelled %' THEN 'CANCELLED' ELSE State END AS Category, 
                COUNT(JobID) AS JobCount
            FROM sqlite_db.allocations
            WHERE Partition != 'jhub' 
              AND State NOT IN ('PENDING', 'RUNNING') 
              AND Start >= ? 
              AND End   <= ? 
              AND JobName != 'interactive'
        """
        params = [start_date, end_date]

        query, params = helpers.build_conditions(query, params, partition_selector, allowed_groups, user_role, current_user)
        
        if current_user:
            query += " AND User = ?"
            params.append(current_user)

        query += " GROUP BY CASE WHEN LOWER(State) LIKE 'cancelled %' THEN 'CANCELLED' ELSE State END"
        
        df = _self.con.execute(query, params).fetchdf()
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

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
        
        df_sorted = df.sort_values(by='JobCount', ascending=False)
        st.dataframe(df_sorted, hide_index=True, use_container_width=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_job_runtime(
        _self,
        start_date,
        end_date,
        scale_efficiency=True,
        partition_selector=None,
        user_role=None,
        current_user=None,
        allowed_groups=None
    ) -> None:
        query = """
            SELECT 
                CAST((End - Start) AS DOUBLE) / 60.0 AS runtime_minutes,
                CAST(CPUTime  AS DOUBLE) AS CPUTime,
                CAST(TotalCPU AS DOUBLE) AS TotalCPU
            FROM sqlite_db.allocations
            WHERE Partition != 'jhub'
              AND State NOT IN ('PENDING', 'RUNNING')
              AND JobName != 'interactive'
              AND Start >= ?
              AND End   <= ?
        """
        params = [start_date, end_date]

        query, params = helpers.build_conditions(
            query, params, partition_selector, allowed_groups, user_role, current_user
        )

        if current_user:
            query += " AND User = ?"
            params.append(current_user)

        df = _self.con.execute(query, params).fetchdf()
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        for col in ("runtime_minutes", "CPUTime", "TotalCPU"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["runtime_minutes", "CPUTime", "TotalCPU"])

        factor = 0.5 if scale_efficiency else 1.0
        df["lost_days_raw"] = ((df["CPUTime"] * factor) - df["TotalCPU"]) / 86400.0

        total_cpu_days = (df["CPUTime"] * factor).sum() / 86400.0
        total_lost_cpu_days = max(df["lost_days_raw"].sum(), 0.0)

        max_runtime = df["runtime_minutes"].max()
        predefined_bins = [0, 2, 5, 10, 20, 60, 120, 240, 480, 1440, 2880, 5760, 11520, 23040]
        bins = [b for b in predefined_bins if b <= max_runtime] + [max_runtime]
        bins = sorted(set(bins))
        if len(bins) < 2:
            bins = [0, max(1, max_runtime)]

        df["runtime_interval"] = pd.cut(df["runtime_minutes"], bins=bins, right=True, include_lowest=True)

        grouped = df.groupby("runtime_interval", observed=True).agg({
            "lost_days_raw": "sum"
        }).reset_index()

        if total_lost_cpu_days <= 0:
            grouped["lost_cpu_days"] = 0.0
        else:
            pos_sum = grouped.loc[grouped["lost_days_raw"] > 0, "lost_days_raw"].sum()
            if pos_sum > 0:
                scale = total_lost_cpu_days / pos_sum
                grouped["lost_cpu_days"] = grouped["lost_days_raw"].clip(lower=0) * scale
            else:
                grouped["lost_cpu_days"] = 0.0

        grouped["lost_cpu_days"] = grouped["lost_cpu_days"].round(1)
        grouped["runtime_interval"] = grouped["runtime_interval"].apply(helpers.format_interval_label)

        st.markdown('Lost CPU Time by Job Time', help='Partition "jhub" and Interactive Jobs are excluded')
        fig = px.pie(grouped, names="runtime_interval", values="lost_cpu_days")
        st.plotly_chart(fig)

        cluster_efficiency = ((total_cpu_days - total_lost_cpu_days) / total_cpu_days * 100.0) if total_cpu_days > 0 else 0.0
        summary_data = {
            "total CPU days booked": f"{int(round(total_cpu_days)):,}",
            "total CPU days lost": f"{int(round(total_lost_cpu_days)):,}",
            "cluster efficiency": f"{cluster_efficiency:.2f}%"
        }
        df2 = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Time in days"]).set_index("Metric")

        st.write("Cluster Efficiency")
        st.dataframe(df2, use_container_width=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_batch_inter(
        _self, start_date, end_date, current_user, user_role, scale_efficiency=True,
        partition_selector=None, allowed_groups=None
    ) -> None:
        if scale_efficiency:
            query = """
                SELECT
                    ROUND(((CPUTime * 0.5) - TotalCPU) / 86400.0, 1) AS lost_cpu_days,
                    JobName
                FROM sqlite_db.allocations
                WHERE Partition != 'jhub' 
                  AND Start >= ? 
                  AND End   <= ?
                  AND JobName != 'interactive'
            """
        else:
            query = """
                SELECT             
                    ROUND((CPUTime - TotalCPU) / 86400.0, 1) AS lost_cpu_days,
                    JobName
                FROM sqlite_db.allocations
                WHERE Partition != 'jhub' 
                  AND Start >= ? 
                  AND End   <= ? 
                  AND JobName != 'interactive'
            """
        
        params = [start_date, end_date]

        query, params = helpers.build_conditions(query, params, partition_selector, allowed_groups, user_role, current_user)

        if current_user:
            query += " AND User = ?"
            params.append(current_user)

        df = _self.con.execute(query, params).fetchdf()
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        conditions = [
            df['JobName'] == 'spawner-jupyterhub',
            df['JobName'] == 'interactive', 
            df['JobName'] != ''
        ]
        choices = ['Jupyterhub', 'Interactive', 'Batch']
        df['Category'] = np.select(conditions, choices, default='None')

        aggregated_df = df.groupby('Category', as_index=False)['lost_cpu_days'].sum()
        aggregated_df['lost_cpu_days'] = aggregated_df['lost_cpu_days'].clip(lower=0)
        
        color_map = {
            'Interactive': 'red',
            'Batch': 'darkcyan',
            'None': 'grey',
            'Jupyterhub': 'orange'
        }

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
