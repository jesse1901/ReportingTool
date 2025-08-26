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
                ::-webkit-scrollbar {
                    width: 10px;
                }
                
                /* Track */
                ::-webkit-scrollbar-track {
                    background: #f1f1f1;
                }
                
                /* Handle */
                ::-webkit-scrollbar-thumb {
                    background: #888;
                }
                
                /* Handle on hover */scroll
                ::-webkit-scrollbar-thumb:hover {
                    background: #555;
                }
            </style>
            </head>
            <body>
            </body>
        </html>
        """

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_all_data(
        _self, 
        current_user: str, 
        user_role: str, 
        number: int = None, 
        partition_selector: str = None, 
        filter_jobid: str = None, 
        filter_user: str = None,
        start_date: str = None,
        end_date: str = None, 
        allowed_groups=None
    ) -> pd.DataFrame:
        """
        Retrieves data from the reportdata table based on user role.
        
        Args:
            current_user: Current user identifier
            user_role: Role of the user ('admin', 'exfel', or 'user')
            number: Limit for number of records to return
            partition_selector: Filter by partition
            filter_jobid: Filter by specific job ID
            filter_user: Filter by specific user
            start_date: Start date for filtering (YYYY-MM-DD format)
            end_date: End date for filtering (YYYY-MM-DD format)
            
        Returns:
            DataFrame with job allocation data
        """
        base_query = """
        SELECT 
            jobID, JobName, User, Account, State, 
            ROUND(Elapsed / 3600, 2) AS Elapsed_hours, 
            Start, End, Partition, NodeList, AllocCPUS,  
            ROUND((CPUTime / 3600), 2) AS CPU_hours, 
            ROUND((TotalCPU / 3600), 2) AS CPU_hours_used, 
            ROUND((CPUTime - TotalCPU) / 3600, 2) AS CPU_hours_lost, 
            ROUND(CPUEff * 100, 2) AS CPUEff, 
            NGPUS AS AllocGPUS, 
            ROUND(GpuUtil * 100, 2) AS GPUEff,
            ROUND((NGPUS * Elapsed) * (1 - GpuUtil) / 3600, 2) AS GPU_hours_lost, 
            Comment, SubmitLine 
        FROM allocations
        """

        conditions = []
        params = []

        # Build WHERE conditions based on filters
        if filter_jobid:
            conditions.append("JobID = ?")
            params.append(filter_jobid)

        if filter_user:
            conditions.append("User = ?")
            params.append(filter_user)

        if user_role == 'user':   # Regular users only see their own data
            conditions.append("User = ?")
            params.append(current_user)

        if user_role == 'admin' and current_user:
            conditions.append("User = ?")
            params.append(current_user)

        if partition_selector:
            conditions.append("Partition = ?")
            params.append(partition_selector)

        if allowed_groups is not None and user_role == 'uhh':
            placeholders = ','.join('?' for _ in allowed_groups)
            conditions.append(f"Account IN ({placeholders})")
            params.extend(allowed_groups)

        if start_date:
            conditions.append("Start >= ?")
            params.append(start_date)

        if end_date:
            conditions.append("End <= ?")
            params.append(end_date)

        # Construct final query
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY End DESC LIMIT ?"
        params.append(int(number))

        return pd.read_sql_query(base_query, _self.con, params=params)



    def frame_user_all(
        self, 
        current_user: str, 
        user_role: str, 
        number: int,
        partition_selector: str, 
        filter_jobid: str, 
        filter_user: str,
        start_date: str,
        end_date: str,
        allowed_groups=None) -> None:

        # Apply custom CSS for scrollbars
        st.markdown(self.SCROLLBAR_CSS, unsafe_allow_html=True)

        col1, _ = st.columns([1, 2])

        with col1:
            st.markdown(
                'User Data',
                help=
"""
There may be delays of a up to one hour when updating the GPU data.  
Furthermore, the hyperthreading option is not applied to this data frame, therefore   
all columns and calculations in this DataFrame contain the hyperthreading cores,  
no matter which option is selected
"""
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
            current_user, user_role, number, partition_selector, filter_jobid, filter_user, start_date, end_date
        )
        
        df = helpers.convert_timestamps_to_berlin_time(df)
        
        # Display interactive dataframe
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
        start_date: str, 
        end_date: str, 
        current_user: str, 
        user_role: str, 
        scale_efficiency: bool, 
        partition_selector: str = None,
        allowed_groups=None
    ) -> None:
        
    
        # Validate date range
        if not (start_date and end_date):
            return
            
        if start_date > end_date:
            st.error("Error: End date must fall after start date.")
            return

        # Base query for regular efficiency calculation
        base_query_normal = """
        SELECT
            eff.User,
            eff.Account,
            COUNT(eff.JobID) AS JobCount,
            ROUND(SUM(eff.cpu_s_reserved - eff.cpu_s_used) / 86400, 1) AS Lost_CPU_days,
            ROUND(SUM(slurm.CPUTime) / 86400, 1) AS cpu_days,
            printf('%2.0f%%', 100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) AS CPUEff,
            ROUND(SUM(eff.Elapsed * eff.NGPUs) / 86400, 1) AS GPU_Days,
            ROUND(SUM((eff.NGPUS * eff.Elapsed) * (1 - eff.GPUeff)) / 86400, 1) AS Lost_GPU_Days,
            iif(SUM(eff.NGPUs), printf("%2.0f%%", 100 * SUM(eff.Elapsed * eff.NGPUs * eff.GPUeff) / SUM(eff.Elapsed * eff.NGPUs)), NULL) AS GPUEff,
            ROUND(SUM(eff.TotDiskRead / 1048576) / SUM(eff.Elapsed), 2) AS read_MiBps,
            ROUND(SUM(eff.TotDiskWrite / 1048576) / SUM(eff.Elapsed), 2) AS write_MiBps
        FROM eff
        JOIN slurm ON eff.JobID = slurm.JobID
        WHERE eff.Start >= ? 
        AND eff.End <= ? 
        AND eff.End IS NOT NULL 
        AND slurm.Partition != 'jhub'
        AND slurm.JobName != 'interactive'
        """

        # Query with hyperthreading scaling applied
        base_query_scaled = """
        SELECT
            eff.User,
            eff.Account,
            COUNT(eff.JobID) AS JobCount,
            ROUND(SUM((eff.cpu_s_reserved / 2) - eff.cpu_s_used) / 86400, 1) AS Lost_CPU_days,
            ROUND(SUM(slurm.CPUTime) / 86400 / 2, 1) AS cpu_days,
            ROUND(iif((100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) * 2 > 100, 100, (100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) * 2), 1) AS CPUEff,
            ROUND(SUM(eff.Elapsed * eff.NGPUs) / 86400, 1) AS GPU_Days,
            ROUND(SUM((eff.NGPUS * eff.Elapsed) * (1 - eff.GPUeff)) / 86400, 1) AS Lost_GPU_Days,
            iif(SUM(eff.NGPUs) > 0, 100 * SUM(eff.Elapsed * eff.NGPUs * eff.GPUeff) / SUM(eff.Elapsed * eff.NGPUs), NULL) AS GPUEff,
            ROUND(SUM(eff.TotDiskRead / 1048576) / SUM(eff.Elapsed), 2) AS read_MiBps,
            ROUND(SUM(eff.TotDiskWrite / 1048576) / SUM(eff.Elapsed), 2) AS write_MiBps
        FROM eff
        JOIN slurm ON eff.JobID = slurm.JobID
        WHERE eff.Start >= ? 
        AND eff.End <= ? 
        AND eff.End IS NOT NULL 
        AND slurm.Partition != 'jhub'
        AND slurm.JobName != 'interactive'
        """

        # Choose query based on scaling preference
        base_query = base_query_scaled if scale_efficiency else base_query_normal
        params = [start_date, end_date]

        if partition_selector:
            base_query += " AND slurm.Partition = ?"
            params.append(partition_selector)

        if user_role == 'user' and current_user:
            base_query += " AND eff.User = ?"
            params.append(current_user)

        if user_role == 'admin' and current_user:
            base_query += " AND eff.User = ?"
            params.append(current_user)

        if allowed_groups:
            placeholders = ','.join('?' for _ in allowed_groups)
            base_query += f" AND eff.Account IN ({placeholders})"
            params.extend(allowed_groups)


        # Execute query
        df = pd.read_sql_query(base_query + " GROUP BY eff.User", _self.con, params=params)
        
        # Ensure Lost_CPU_days is not negative
        df['Lost_CPU_days'] = df['Lost_CPU_days'].clip(lower=0)
        
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        # For regular users, transpose the data for better display
        if user_role == 'user':
            df = df.T.reset_index()
            df.columns = ["Metric", "Value"]
        
        # Display results
        st.markdown("<div style='height: 81px;'></div>", unsafe_allow_html=True)
        st.markdown(
            "Data Grouped by User", 
            help='Partition "jhub" and Interactive Jobs are excluded'
        )
        st.dataframe(df, use_container_width=False)

