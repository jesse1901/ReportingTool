import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import sqlite3
import plotly.express as px
import altair as alt
import subprocess
import pytz

 #CHECK : hover Account, cataldi??? , upex into exfel, button change view, 
class timeUtils:
    def timestring_to_seconds(timestring):
        if pd.isna(timestring) or timestring == '0' or timestring == 0 or timestring.strip() == '':
            return 0

        if isinstance(timestring, float):
            timestring = str(int(timestring)) 

        timestring = str(timestring).strip()

        if 'T' in timestring:
            days_part, time_part = timestring.split('T')
        else:
            days_part, time_part = '0', timestring

        try:
            days = int(days_part.strip()) if days_part.strip() else 0
        except ValueError:
            days = 0

        # Convert time part (HH:MM:SS)
        time_parts = time_part.split(':')
        try:
            hours = int(time_parts[0].strip()) if len(time_parts) > 0 and time_parts[0].strip() else 0
        except ValueError:
            hours = 0
        try:
            minutes = int(time_parts[1].strip()) if len(time_parts) > 1 and time_parts[1].strip() else 0
        except ValueError:
            minutes = 0
        try:
            seconds = int(time_parts[2].strip()) if len(time_parts) > 2 and time_parts[2].strip() else 0
        except ValueError:
            seconds = 0

        # Calculate total seconds
        total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
        return total_seconds



    def seconds_to_timestring(total_seconds):
        if pd.isna(total_seconds):
            return None
        if isinstance(total_seconds, float):
            total_seconds = int(total_seconds)

        if isinstance(total_seconds, int) and total_seconds >= 0: 

            td = timedelta(seconds=total_seconds)

            days = td.days
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            seconds = round(seconds)  

            timestring = f"{days}T {hours}:{minutes}:{seconds}"
            return timestring
        else:
            return None

    def format_interval_label(interval):
        min_time = interval.left
        max_time = interval.right

        def format_time(minutes):
            days = int(minutes // 1440)  
            hours = int((minutes % 1440) // 60)
            mins = int(minutes % 60)

            if days > 0 and hours > 0:
                return f"{days}d {hours}h"
            elif days > 0:
                return f"{days}d"
            elif hours > 0 and mins > 0:
                return f"{hours}h {mins}m"
            elif hours > 0:
                return f"{hours}h"
            else:
                return f"{mins}m"

        min_time_str = format_time(min_time)
        max_time_str = format_time(max_time)
        return f"{min_time_str} - {max_time_str}"


def get_job_script(_self, jobid):
    command = ["sacct", "-B", "-j", jobid]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        st.code(output)
    except subprocess.CalledProcessError as e:
        st.error(f"Error details: {e}")
        

def readable_with_commas(value):
    if value >= 100_000:
        return f"{value / 1_000_000:.3f}M"
    elif value <= 100_000 and value >= 1_000:
        return f"{value / 1_000:.3f}K"
    else:
        return f"{value:,}"

class CreateFigures:

    def __init__(_self, con):

        _self.con = sqlite3.connect('max-reports-slurm.sqlite3')
        _self.color_map = {
         'CANCELLED': "#803df5", #'#1f77b4 ',    # Light Blue
         'COMPLETED':  "#5ce488",  ##17becf ',    # Light Sky Blue
        'TIMEOUT': "#1c83e1", #'#d62728 ',     # red
        'FAILED': "#ff2b2b",      # Pink
        'PREEMPTED': '#ffe312',     # Light Green
        'NODE_FAIL': '#566573'
    }

    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_all_data(_self, current_user, user_role, number=None, partition_selector=None, filter_jobid=None, filter_user=None):
        """
        Retrieves data from the reportdata table based on user role.
        """
        query = """SELECT jobID, JobName, User, Account, State, 
        ROUND(Elapsed / 60,1) AS Elapsed_hours, 
        Start, End,  Partition, NodeList, AllocCPUS,  
        ROUND((CPUTime / 3600),2) AS CPU_hours, ROUND((TotalCPU / 3600),2) AS CPU_hours_used, 
        ROUND((CPUTime - TotalCPU)/3600,2) AS CPU_hours_lost, ROUND(CPUEff*100, 2) AS CPUEff, 
        NGPUS AS AllocGPUS, ROUND(GpuUtil*100,2) AS GPUEff,
        ROUND((NGPUS * Elapsed) * (1 - GpuUtil) / 3600, 2) AS GPU_hours_lost, Comment, SubmitLine 
        FROM allocations """



        conditions = []
        params = []

        # Add filter for JobID
        if filter_jobid:
            conditions.append("JobID = ?")
            params.append(filter_jobid)

        # Add filter for User
        if filter_user:
            conditions.append("User = ?")
            params.append(filter_user)

        # Add filter based on user role
        if user_role == "admin":
            pass  # Admins see all data
        elif user_role == "exfel":
            pass
            #conditions.append("Account IN ('exfel', 'upex')")
        else:
            conditions.append("User = ?")
            params.append(current_user)

        # Add filter for Partition
        if partition_selector:
            conditions.append("Partition = ?")
            params.append(partition_selector)

        # Combine conditions into a WHERE clause
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add ORDER BY and LIMIT
        query += " ORDER BY End DESC LIMIT ?"
        params.append(int(number))

        return pd.read_sql_query(query, _self.con, params=params)
    
    def frame_user_all(_self, current_user, user_role, number ,partition_selector, filter_jobid, filter_user) -> None:
        """
        Displays all job data from the reportdata table in the Streamlit app.
        """
        st.markdown("""
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

                            /* Handle on hover */
                            ::-webkit-scrollbar-thumb:hover {
                            background: #555;
                            }
                    </style>
                    </head>
                    <body>
                    </body>
                </html>
            """, unsafe_allow_html=True)

        col1, _ = st.columns([1,2])


        with col1:
            st.markdown('User Data',help= """There may be delays of a few hours when updating the GPU data.  
                                            Furthermore, the hyperthreading option is not applied to this data frame, therefore   
                                            all columns and calculations in this Daraframe contain the hyperthreading cores,  
                                            no matter which option is selected""")
            with st.expander("🛈    Job script"):  # Box standardmäßig ausgeklappt
                
                st.markdown(
                    """
                    <div style="text-align: left;">
                        <span style="font-size: 16px;">Click on the first column of the row to display the job script </span>
                        <br>
                        <span style="font-size: 28px;">🠯</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        df = CreateFigures.fetch_all_data(_self, current_user, user_role, number, partition_selector, filter_jobid, filter_user)

        berlin_tz = pytz.timezone('Europe/Berlin')

        # Convert Unix timestamps to datetime and localize to UTC
        df['Start'] = pd.to_datetime(df['Start'], unit='s', errors='coerce').dt.tz_localize('UTC')
        df['End'] = pd.to_datetime(df['End'], unit='s', errors='coerce').dt.tz_localize('UTC')

        # Convert to Berlin time, accounting for daylight saving time
        df['Start'] = df['Start'].dt.tz_convert(berlin_tz)
        df['End'] = df['End'].dt.tz_convert(berlin_tz)

        # Format datetime columns to display time without offset
        df['Start'] = df['Start'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['End'] = df['End'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        
        event = st.dataframe(df, on_select="rerun",selection_mode="single-row" ,key="user_all",    
                            use_container_width=True, hide_index=True)

        row = event.selection.rows
        filtered_df = df.iloc[row]
        
        if len(filtered_df) > 0:
            get_job_script(_self, jobid=filtered_df.JobID.iloc[0])

    
    @st.cache_data(ttl=3600, show_spinner=False)
    def frame_group_by_user(_self, start_date, end_date, current_user, user_role, scale_efficiency, partition_selector=None) -> None:

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return 
            base_query = """
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
            params = [start_date, end_date]

            if scale_efficiency: 
                base_query = """
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

            if user_role == 'admin':
                pass
            elif user_role == 'exfel':
                pass
                #base_query += " AND eff.Account IN ('exfel', 'upex')"
            elif user_role == 'user':
                base_query += " AND eff.User = ?"
                params.append(current_user)

            if partition_selector:
                base_query += " AND slurm.Partition = ?"
                params.append(partition_selector)

            df = pd.read_sql_query(base_query + " GROUP BY eff.User", _self.con, params=params)
            
            df['Lost_CPU_days'] = df['Lost_CPU_days'].clip(lower=0)
            if df.empty:
                st.warning("No data available for the selected date range or partition.")
                return

            if user_role == 'user':
                df = df.T.reset_index()
                df.columns = ["Metric", "Value"]
            
            
            st.markdown("<div style='height: 81px;'></div>", unsafe_allow_html=True)
            st.markdown("Data Grouped by User", help='Partition "jhub" and Interactive Jobs are excluded')
            st.dataframe(df)


    @st.cache_data(ttl=3600, show_spinner=False)
    def bar_char_by_user(_self , start_date, end_date, current_user, user_role, number=None,scale_efficiency=True, partition_selector=None) -> None:
        st.markdown('Total Lost CPU-Time per User', help='Partition "jhub" and Interactive Jobs are excluded')

        # Set parameters for the SQL query
        params = [start_date, end_date]

        # Select query based on `scale_efficiency`
        if scale_efficiency:
            query = """
                SELECT 
                    eff.User,
                    COUNT(eff.JobID) AS job_count,
                    ROUND(SUM((eff.cpu_s_reserved / 2) - eff.cpu_s_used) / 86400, 1) AS lost_cpu_days,
                    eff.Account
                FROM eff
                JOIN slurm ON eff.JobID = slurm.JobID
                WHERE eff.Start >= ? 
                AND eff.Start IS NOT NULL
                AND eff.End <= ? 
                AND eff.End IS NOT NULL 
                AND slurm.Partition != 'jhub'
                AND slurm.JobName != 'interactive'
            """
        else:
            query = """
                SELECT 
                    eff.User,
                    ROUND(SUM((eff.cpu_s_reserved - eff.cpu_s_used) / 86400), 1) AS lost_cpu_days,
                    COUNT(eff.JobID) AS job_count,
                    ROUND(SUM(eff.Elapsed * eff.NCPUS) / 86400, 1) AS total_cpu_days,
                    eff.Account
                FROM eff
                JOIN slurm ON eff.JobID = slurm.JobID
                WHERE eff.Start >= ? 
                AND eff.End <= ? 
                AND eff.End IS NOT NULL 
                AND slurm.Partition != 'jhub'
                AND slurm.JobName != 'interactive'
            """

        if user_role == 'exfel':
            pass
            #query += " AND eff.Account IN ('exfel', 'upex')"
        elif user_role == 'user':
            query += " AND eff.User = ?"
            params.append(current_user)

        if partition_selector:
            query += " AND slurm.Partition = ?"
            params.append(partition_selector)

        query += " GROUP BY eff.User ORDER BY lost_cpu_days DESC"

        result_df = pd.read_sql_query(query, _self.con, params=params)
        result_df['lost_cpu_days'] = result_df['lost_cpu_days'].clip(lower=0)
        result_df = result_df.sort_values(by='lost_cpu_days', ascending=False).head(number)

        max_lost_time = result_df['lost_cpu_days'].max()
        tick_vals = np.nan_to_num(np.linspace(0, max_lost_time, num=10), nan=0)
        tick_text = [(int(val)) for val in tick_vals]

        fig = px.bar(result_df, x='User', y='lost_cpu_days', hover_data=['job_count', 'Account'])

        fig.update_layout(
            xaxis=dict(
                title='User',
                tickangle=-45  # Rotate x-axis labels for better readability
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
    def job_counts_by_log2(_self, start_date, end_date, number, partition_selector) -> None:
        st.markdown('Job Count by Job Time', help='Partition "jhub" and Interactive Jobs are excluded')
        
        min_runtime = max(0, number)        
        
        query = """
            SELECT (End - Start) / 60 AS runtime_minutes
            FROM allocations
            WHERE Partition != 'jhub'
            AND (End - Start) / 60 >= ? AND Start >= ? AND End <= ? AND JobName != 'interactive'
        """
        params = [number, start_date, end_date]

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)

        df = pd.read_sql_query(query, _self.con, params=params)
        
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        df['runtime_minutes'] = df['runtime_minutes'].fillna(0)
        max_runtime = df['runtime_minutes'].max()
        
        bins = [0,2,5,10,20,60,120,240,480,1440, 2880,5760,11520, 23040, max_runtime]
        bins = [b for b in bins if b >= min_runtime and b <= max_runtime]  
        bins = sorted(set(bins))
        
        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, include_lowest=True, right=False)
        df['runtime_interval'] = df['runtime_interval'].apply(timeUtils.format_interval_label)
        
        job_counts = df['runtime_interval'].value_counts().sort_index()
        job_counts_df = job_counts.reset_index()
        job_counts_df.columns = ['runtime_interval', 'job_count']

        # st.write(alt.Chart(job_counts_df).mark_bar().encode(
        #     x=alt.X('runtime_interval:O', sort=None),
        #     y=alt.Y('job_count:Q', title=None),
        # ).properties(
        #     width=550, 
        #     height=450  
        # ).configure_axis(
        #     labelFontSize=15
        # ))


        # assume job_counts_df has columns "runtime_interval" and "job_count"
        fig = px.bar(
            job_counts_df,
            x="runtime_interval",
            y="job_count",
            width=550,
            height=450,
        )

        # remove the y-axis title and bump up the x-axis label font
        fig.update_layout(
            xaxis_tickangle=-90,   # oder 90, je nach Ausrichtung
            yaxis_title=None,
            xaxis=dict(
                title=None,
                tickfont=dict(size=15),
            ),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig, use_container_width=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_job_runtime(_self, start_date, end_date, scale_efficiency=True, partition_selector=None) -> None:
        if scale_efficiency:
            query = """
                SELECT
                    (End - Start) / 60 AS runtime_minutes,
                   ROUND(((CPUTime / 2) - TotalCPU) / 86400, 1) AS lost_cpu_days

                   FROM allocations
                WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """

            query2 = """
            SELECT 
                CAST(ROUND(SUM(CPUTime / 86400) / 2) AS INTEGER) AS total_cpu_days,
                CAST(ROUND(SUM((CPUTime / 2) - TotalCPU) / 86400, 1) AS INTEGER) AS lost_cpu_days

            FROM allocations
            WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """
            params=[start_date, end_date]

        else:
            query = """
                SELECT
                    (End - Start) / 60 AS runtime_minutes,
                    ROUND((CPUTime - TotalCPU) / 86400, 1) AS lost_cpu_days         
                FROM allocations
                WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """

            query2 = """
                SELECT 
                CAST(ROUND(SUM(CPUTime / 86400)) AS INTEGER) AS total_cpu_days, 
                CAST(ROUND(SUM((CPUTime- TotalCPU) / 86400)) AS INTEGER) AS lost_cpu_days
                FROM allocations
                WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """

            params = [start_date, end_date]

        if partition_selector:
            query += " AND Partition = ?"
            query2 += " AND Partition = ?"
            params.append(partition_selector)

        df = pd.read_sql_query(query, _self.con, params=params)
        df['lost_cpu_days'] = df['lost_cpu_days'].clip(lower=0)

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return
        
        max_runtime = df['runtime_minutes'].max()
        bins = [0,2,5,10,20,60,120,240,480,1440, 2880,5760,11520, 23040, max_runtime]
        bins = [b for b in bins if b <= max_runtime]
        bins = sorted(set(bins))

        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins)

        cpu_time_by_interval = df.groupby('runtime_interval', observed=True)['lost_cpu_days'].sum().reset_index()
        cpu_time_by_interval['runtime_interval'] = cpu_time_by_interval['runtime_interval'].apply(timeUtils.format_interval_label)

        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='lost_cpu_days')
        
        st.markdown('Lost CPU Time by Job Runtime Interval', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)

        df2 = pd.read_sql_query(query2, _self.con, params=params)
        df2['lost_cpu_days'] = df2['lost_cpu_days'].clip(lower=0)
        if df2.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        df2['cluster_efficiency'] = (df2['total_cpu_days'] - df2['lost_cpu_days']) / df2['total_cpu_days'] * 100

        df2 = df2.rename(columns={
            'total_cpu_days': 'total CPU days booked',
            'lost_cpu_days': 'total CPU days lost',
            'cluster_efficiency': 'cluster efficiency'
        })
        df2 = df2.transpose()
        df2.columns = ['Time in days']
        
        df2['Time in days'] = df2['Time in days'].astype(object)
        df2.loc['cluster efficiency', 'Time in days'] = f"{df2.loc['cluster efficiency', 'Time in days']:.2f}%"
        df2.loc['total CPU days booked', 'Time in days'] = f"{int(df2.loc['total CPU days booked', 'Time in days']):,}"
        df2.loc['total CPU days lost', 'Time in days'] = f"{int(df2.loc['total CPU days lost', 'Time in days']):,}"
        st.write('Cluster Efficiency')
        st.dataframe(df2)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_batch_inter(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None) -> None:
        if scale_efficiency:
            query = """
                SELECT
                    ROUND(((CPUTime / 2) - TotalCPU) / 86400, 1) AS lost_cpu_days,
                    JobName, 
                    Partition 
                FROM allocations 
                WHERE Partition != 'jhub' 
                AND Start >= ? 
                AND End <= ?
                AND JobName != 'interactive'
            """
            params = [start_date, end_date]
        else:
            query = """
                SELECT             
                    ROUND((CPUTime - TotalCPU) / 86400) AS lost_cpu_days,
                    JobName, Partition 
                FROM allocations 
                WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """
            params = [start_date, end_date]

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)
        
        if user_role == 'admin':
            pass
        elif user_role == 'exfel':
            pass
            #query += " AND Account IN ('exfel', 'upex')"
        elif user_role == 'user':
            query += " AND User = ?"
            params.append(current_user)

        df = pd.read_sql_query(query, _self.con, params=params)
        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return

        df['Category'] = df.apply(
            lambda row: 'Jupyterhub' if row['JobName'] == 'spawner-jupyterhub'
            else 'Interactive' if row['JobName'] == 'interactive'
            else 'Batch' if row['JobName'] != ''
            else 'None',
            axis=1
        )

        aggregated_df = df.groupby('Category', as_index=False).agg({'lost_cpu_days': 'sum'})
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
        st.dataframe(aggregated_df, hide_index=True)

    @st.cache_data(ttl=3600, show_spinner=False)
    def scatter_chart_data_cpu_gpu_eff(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None):
        st.markdown('CPU & GPU Efficiency by Job Duration', help='Partition "jhub" and Interactive Jobs are excluded')
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        query = """
            SELECT JobID, User, COALESCE(GpuUtil, 0) AS GpuUtil, 
                (CPUEff * 100) AS CPUEff, 
                ROUND((CPUTime - TotalCPU) / 86400, 1) AS lost_cpu_days, 
                Elapsed, AllocCPUS, State, Partition
            FROM allocations
            WHERE Start >= ? AND End <= ? 
            AND Partition != 'jhub'
            AND JobName != 'interactive'
        """
        params = [start_date, end_date]

        if user_role == "admin":
            query += "AND Elapsed > 100 "
        elif user_role == "exfel":
            pass
            #query += "AND Account IN ('exfel', 'upex')"
        elif user_role == "user":
            query += "AND User == ?"
            params.append(current_user)

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)

        df = pd.read_sql_query(query + " ORDER BY Elapsed ASC;", _self.con, params=params)

        if df.empty:
            st.warning("No data available for the selected date range or partition.")
            return
        
        df['GpuUtil'] = df['GpuUtil'] * 100
        
        df = df.dropna(subset=['Elapsed'])
        df['Elapsed'] = pd.to_numeric(df['Elapsed'], errors='coerce').astype(int).apply(timeUtils.seconds_to_timestring)

        df['CPUEff'] = df['CPUEff'].clip(upper=100)

        if scale_efficiency:
            df['CPUEff'] = df['CPUEff'].where(df['CPUEff'] > 100, df['CPUEff'] * 2).clip(upper=100)

        df['lost_cpu_days'] = df['lost_cpu_days'].fillna(0)

        df['hover_text'] = df.apply(
            lambda row: f"JobID: {row['JobID']}<br>User: {row['User']}<br>Lost CPU days: {row['lost_cpu_days']}<br>CPU Efficiency: {row['CPUEff']}", axis=1
        )

        custom_color_scale = [
            [0.0, "CornflowerBlue"],  
            [0.25, "cyan"],
            [0.5, "green"],     
            [0.75, "orange"],  
            [1.0, "red"],       
        ]

        fig = px.scatter(
            df, 
            x='Elapsed', 
            y='CPUEff', 
            range_color=[0,100],
            color='GpuUtil',
            color_continuous_scale=custom_color_scale,
            hover_data={'JobID': True, 'User': True, 'lost_cpu_days': True, 'CPUEff': True},
            labels={'Elapsed': 'Elapsed Time', 'CPUEff': 'CPU Efficiency (%)'}
        )
        
        fig.update_traces(marker=dict(size=2.5))
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            xaxis_title='Elapsed Time',
            yaxis_title='CPU Efficiency (%)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        st.plotly_chart(fig, theme=None)



    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_session_state(_self, start_date, end_date, current_user, user_role, scale_efficiency=True, partition_selector=None):
        if scale_efficiency:
            query = """
                SELECT 
                    IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
                    CAST(ROUND(SUM((CPUTime / 2) - TotalCPU) / 86400, 1) AS INTEGER) AS lost_cpu_days
                FROM allocations
                WHERE Partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """
            params = [start_date, end_date]
        else:
            query = """
                SELECT 
                    IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
                    CAST(ROUND(SUM((CPUTime - TotalCPU) / 86400)) AS INTEGER) AS lost_cpu_days
                FROM allocations
                WHERE Partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """
            params = [start_date, end_date]

        if user_role == "admin":
            pass
        elif user_role == "exfel":
            pass
            #query += " AND Account IN ('exfel', 'upex')"
        elif user_role == "user":
            query += " AND User == ?"
            params.append(current_user)

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)

        df = pd.read_sql_query(query + " GROUP BY Category", _self.con, params=params)
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
        
        df_grouped = df.groupby('Category').agg({
            'lost_cpu_days': 'sum' 
        }).reset_index()

        df_grouped = df_grouped.sort_values(by='lost_cpu_days', ascending=False)

        st.dataframe(df_grouped, hide_index=True)


    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_job_count(_self, start_date, end_date, current_user, user_role, partition_selector=None):
        query = """
            SELECT
                IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category, COUNT(JobID) AS JobCount
            FROM allocations
            WHERE Partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' AND Start >= ? AND End <= ? AND JobName != 'interactive'
        """
        params = [start_date, end_date]

        if user_role == 'admin':    
            pass
        elif user_role == 'exfel':
            pass
            #query += " AND Account IN ('exfel', 'upex')"
        elif user_role == 'user':
            query += " AND User = ?"
            params.append(current_user)

        if partition_selector:
            query += " AND Partition = ?"
            params.append(partition_selector)
            
        df = pd.read_sql_query(query + " GROUP BY State", _self.con, params=params)

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
        
        df_grouped = df.groupby('Category').agg({
            'JobCount': 'sum' 
        }).reset_index()

        df_grouped = df_grouped.sort_values(by='JobCount', ascending=False)

        st.dataframe(df_grouped, hide_index=True)
