import streamlit as st
import pandas as pd
from helpers import helpers

class DataFrames:
    def __init__(self, connection):
        self.con = connection

        self.SCROLLBAR_CSS = """
        <html>
            <head>
            <style>
                ::-webkit-scrollbar { width: 10px; }
                ::-webkit-scrollbar-track { background: #f1f1f1; }
                ::-webkit-scrollbar-thumb { background: #888; }
                ::-webkit-scrollbar-thumb:hover { background: #555; }
            </style>
            </head>
            <body></body>
        </html>
        """

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_all_data(
        _self, 
        current_user, 
        user_role, 
        number = None, 
        partition_selector = None, 
        filter_jobid = None, 
        filter_user = None,
        start_date = None,
        end_date = None, 
        allowed_groups=None
    ) -> pd.DataFrame:
        """
        Retrieves data from the allocations table based on filters.
        """
        base_query = """
        SELECT 
            JobID, JobName, User, Account, State, 
            ROUND(Elapsed / 3600.0, 2) AS Elapsed_hours, 
            Start, End, Partition, NodeList, AllocCPUS,  
            ROUND((CPUTime / 3600.0), 2) AS CPU_hours, 
            ROUND((TotalCPU / 3600.0), 2) AS CPU_hours_used, 
            ROUND((CPUTime - TotalCPU) / 3600.0, 2) AS CPU_hours_lost, 
            ROUND(CPUEff * 100.0, 2) AS CPUEff, 
            NGPUS AS AllocGPUS, 
            ROUND(GpuUtil * 100.0, 2) AS GPUEff,
            ROUND((NGPUS * Elapsed) * (1 - GpuUtil) / 3600.0, 2) AS GPU_hours_lost, 
            Comment, SubmitLine 
        FROM sqlite_db.allocations
        """

        conditions = []
        params = []

        if filter_jobid:
            conditions.append("JobID = ?")
            params.append(filter_jobid)

        if filter_user:
            conditions.append("User = ?")
            params.append(filter_user)

        # Regular users only see their own data (keeping your original logic)
        if current_user:
            conditions.append("User = ?")
            params.append(current_user)
      
        if partition_selector:
            placeholders = ",".join(["?"] * len(partition_selector))
            conditions.append(f"Partition IN ({placeholders})")
            params.extend(partition_selector)

        if allowed_groups is not None and user_role == 'uhh':
            placeholders = ",".join(["?"] * len(allowed_groups))
            conditions.append(f"Account IN ({placeholders})")
            params.extend(allowed_groups)

        if start_date:
            conditions.append("Start >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("End <= ?")
            params.append(end_date)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY End DESC LIMIT ?"
        params.append(int(number) if number is not None else 1000)

        # DuckDB execution
        return _self.con.execute(base_query, params).fetchdf()

    def frame_user_all(
        self, 
        current_user, 
        user_role, 
        number,
        partition_selector = None, 
        filter_jobid = None, 
        filter_user = None,
        start_date = None,
        end_date =  None,
        allowed_groups=None
    ) -> None:
        # Apply custom CSS for scrollbars
        st.markdown(self.SCROLLBAR_CSS, unsafe_allow_html=True)

        col1, _ = st.columns([1, 2])

        with col1:
            st.markdown(
                'User Data',
                help=(
                    "There may be delays of up to one hour when updating the GPU data.\n\n"
                    "Furthermore, the hyperthreading option is not applied to this data frame, therefore\n"
                    "all columns and calculations in this DataFrame contain the hyperthreading cores,\n"
                    "no matter which option is selected"
                )
            )
            with st.expander("🛈 Job script"):
                st.markdown(
                    """
                    <div style="text-align: left;">
                        <span style="font-size: 16px;">Click on the first column of the row to display the job script</span>
                        <br>
                        <span style="font-size: 28px;">🠯</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Fetch and process data
        df = self.fetch_all_data(
            current_user, user_role, number, partition_selector, filter_jobid,
            filter_user, start_date, end_date, allowed_groups=allowed_groups
        )
        df = helpers.convert_timestamps_to_berlin_time(df)
        
        # Interactive dataframe
        event = st.dataframe(
            df, 
            on_select="rerun",
            selection_mode="single-row",
            key="user_all",    
            use_container_width=False, 
            hide_index=True
        )

        # Handle row selection for job script display
        selected_rows = event.selection.rows
        if selected_rows:
            filtered_df = df.iloc[selected_rows]
            if len(filtered_df) > 0:
                helpers.get_job_script(jobid=filtered_df.JobID.iloc[0])

    @st.cache_data(ttl=3600, show_spinner=False)
    def frame_group_by_user(
        _self, 
        start_date, 
        end_date, 
        current_user, 
        user_role, 
        scale_efficiency, 
        partition_selector = None,
        allowed_groups=None
    ) -> None:
        # Validate date range
        if not (start_date and end_date):
            return
        if start_date > end_date:
            st.error("Error: End date must fall after start date.")
            return

        # Base query (normal)
        base_query_normal = """
        SELECT
            eff.User,
            eff.Account,
            COUNT(eff.JobID) AS JobCount,
            ROUND(SUM(eff.cpu_s_reserved - eff.cpu_s_used) / 86400.0, 1) AS Lost_CPU_days,
            ROUND(SUM(slurm.CPUTime) / 86400.0, 1) AS cpu_days,
            printf('%2.0f%%',
                   100.0 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / NULLIF(SUM(eff.Elapsed * eff.NCPUS), 0)
            ) AS CPUEff,
            ROUND(SUM(eff.Elapsed * eff.NGPUs) / 86400.0, 1) AS GPU_Days,
            ROUND(SUM((eff.NGPUS * eff.Elapsed) * (1 - eff.GPUeff)) / 86400.0, 1) AS Lost_GPU_Days,
            CASE WHEN SUM(eff.NGPUS) > 0 THEN
                 printf('%2.0f%%',
                        100.0 * SUM(eff.Elapsed * eff.NGPUS * eff.GPUeff) / NULLIF(SUM(eff.Elapsed * eff.NGPUS), 0)
                 )
                 ELSE NULL
            END AS GPUEff,
            ROUND(SUM(eff.TotDiskRead / 1048576.0) / NULLIF(SUM(eff.Elapsed), 0), 2) AS read_MiBps,
            ROUND(SUM(eff.TotDiskWrite / 1048576.0) / NULLIF(SUM(eff.Elapsed), 0), 2) AS write_MiBps
        FROM sqlite_db.eff AS eff
        JOIN sqlite_db.slurm AS slurm ON eff.JobID = slurm.JobID
        WHERE eff.Start >= ? 
          AND eff.End   <= ? 
          AND eff.End IS NOT NULL 
          AND slurm.Partition != 'jhub'
          AND slurm.JobName != 'interactive'
        """

        # Base query (scaled for hyperthreading)
        base_query_scaled = """
        SELECT
            eff.User,
            eff.Account,
            COUNT(eff.JobID) AS JobCount,
            ROUND(SUM((eff.cpu_s_reserved / 2.0) - eff.cpu_s_used) / 86400.0, 1) AS Lost_CPU_days,
            ROUND(SUM(slurm.CPUTime) / (86400.0 * 2.0), 1) AS cpu_days,
            printf('%2.0f%%',
                   LEAST(100.0,
                         2.0 * (100.0 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / NULLIF(SUM(eff.Elapsed * eff.NCPUS), 0))
                   )
            ) AS CPUEff,
            ROUND(SUM(eff.Elapsed * eff.NGPUs) / 86400.0, 1) AS GPU_Days,
            ROUND(SUM((eff.NGPUS * eff.Elapsed) * (1 - eff.GPUeff)) / 86400.0, 1) AS Lost_GPU_Days,
            CASE WHEN SUM(eff.NGPUS) > 0 THEN
                 printf('%2.0f%%',
                        100.0 * SUM(eff.Elapsed * eff.NGPUS * eff.GPUeff) / NULLIF(SUM(eff.Elapsed * eff.NGPUS), 0)
                 )
                 ELSE NULL
            END AS GPUEff,
            ROUND(SUM(eff.TotDiskRead / 1048576.0) / NULLIF(SUM(eff.Elapsed), 0), 2) AS read_MiBps,
            ROUND(SUM(eff.TotDiskWrite / 1048576.0) / NULLIF(SUM(eff.Elapsed), 0), 2) AS write_MiBps
        FROM sqlite_db.eff AS eff
        JOIN sqlite_db.slurm AS slurm ON eff.JobID = slurm.JobID
        WHERE eff.Start >= ? 
          AND eff.End   <= ? 
          AND eff.End IS NOT NULL 
          AND slurm.Partition != 'jhub'
          AND slurm.JobName != 'interactive'
        """

        base_query = base_query_scaled if scale_efficiency else base_query_normal
        params = [start_date, end_date]

        if partition_selector:
            placeholders = ",".join(["?"] * len(partition_selector))
            base_query += f" AND slurm.Partition IN ({placeholders})"
            params.extend(partition_selector)

        if current_user:
            base_query += " AND eff.User = ?"
            params.append(current_user)

        if allowed_groups:
            placeholders = ",".join(["?"] * len(allowed_groups))
            base_query += f" AND eff.Account IN ({placeholders})"
            params.extend(allowed_groups)

        # Execute with DuckDB
        df = _self.con.execute(base_query + " GROUP BY eff.User, eff.Account", params).fetchdf()

        # Ensure non-negative
        if 'Lost_CPU_days' in df.columns:
            df['Lost_CPU_days'] = df['Lost_CPU_days'].clip(lower=0)

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        # For regular users, transpose for display
        if user_role == 'user':
            df = df.T.reset_index()
            df.columns = ["Metric", "Value"]
        
        st.markdown("<div style='height: 81px;'></div>", unsafe_allow_html=True)
        st.markdown("Data Grouped by User", help='Partition "jhub" and Interactive Jobs are excluded')
        st.dataframe(df, use_container_width=False)
