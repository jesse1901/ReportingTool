import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta, date
import numpy as np
import sqlite3
import toml
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import asdict
from config import get_config
import altair as alt
import subprocess


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


    #     _self.color_map = {
    #      'CANCELLED': "#FF7F50 ", #'#1f77b4 ',    # Light Blue
    #      'COMPLETED':  " #117a65",#'#17becf ',    # Light Sky Blue
    #     'TIMEOUT': "#FFBF00", #'#d62728 ',     # red
    #     'FAILED': '#c0392b',      # Pink
    #     'PREEMPTED': '#2ca02c',     # Light Green
    #     'NODE_FAIL': ' #566573 '
    # }
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_all_data(_self, current_user, user_role, number):
        """
        Retrieves data from the reportdata table based on user role.
        """
        st.write("All Data")
        if user_role == "admin":
            query = """SELECT jobID, JobName, User, Account, State, Elapsed, Start, End, SubmitLine, Partition, NodeList, AllocCPUS,  ROUND((CPUTime / 3600),2) AS CPU_hours, 
            ROUND((TotalCPU / 3600),2) AS CPU_hours_used, ROUND((CPUTime- TotalCPU)/3600,2) AS CPU_hours_lost,
            CPUEff, NGPUS AS AllocGPUS, GpuUtil AS GPUEff, ROUND((NGPUS * Elapsed) * (1 - GpuUtil) / 3600, 2) AS GPU_hours_lost 
            FROM allocations ORDER BY End DESC LIMIT ?"""
            params = (int(number),)
        elif user_role == "exfel":
            query = """SELECT jobID, JobName, User, Account, State, Elapsed, Start, End, SubmitLine, Partition, NodeList, AllocCPUS,  ROUND((CPUTime / 3600),2) AS CPU_hours, 
            ROUND((TotalCPU / 3600),2) AS CPU_hours_used, ROUND((CPUTime- TotalCPU)/3600,2) AS CPU_hours_lost,
            CPUEff, NGPUS AS AllocGPUS, GpuUtil AS GPUEff, ROUND((NGPUS * Elapsed) * (1 - GpuUtil) / 3600, 2) AS GPU_hours_lost 
            FROM allocations WHERE Account = 'exfel' ORDER BY End DESC LIMIT ?"""
            params = (int(number), )
        else:
            query = """SELECT jobID, JobName, User, Account, State, Elapsed, Start, End, SubmitLine, Partition, NodeList, AllocCPUS,  ROUND((CPUTime / 3600),2) AS CPU_hours, 
            ROUND((TotalCPU / 3600),2) AS CPU_hours_used, ROUND((CPUTime- TotalCPU)/3600,2) AS CPU_hours_lost,
            CPUEff, NGPUS AS AllocGPUS, GpuUtil AS GPUEff, ROUND((NGPUS * Elapsed) * (1 - GpuUtil) / 3600, 2) AS GPU_hours_lost 
            FROM allocations WHERE User = ? LIMIT ?"""
            params = current_user, int(number)

        return pd.read_sql_query(query, _self.con, params=params)

    def frame_user_all(_self, current_user, user_role, number, *args) -> None:
        """
        Displays all job data from the reportdata table in the Streamlit app.
        """
        col1, _ = st.columns([1,2])

        with col1:
            with st.expander("🛈    Job script"):  # Box standardmäßig ausgeklappt
                
                st.markdown(
                    """
                    <div style="text-align: left;">
                        <span style="font-size: 16px;">CLICK on the row to display the job script </span>
                        <br>
                        <span style="font-size: 28px;">🠯</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        df = CreateFigures.fetch_all_data(_self, current_user, user_role, number)

        df['Start'] = pd.to_datetime(df['Start'], unit='s', errors='coerce')
        df['End'] = pd.to_datetime(df['End'], unit='s', errors='coerce', )

        # Add 1 hour to the datetime values
        df['Start'] = df['Start'] + pd.to_timedelta(1, unit='h')
        df['End'] = df['End'] + pd.to_timedelta(1, unit='h')
        
        event = st.dataframe(df, on_select="rerun",selection_mode="single-row" ,key="user_all",    
                            use_container_width=True, hide_index=True)

        row = event.selection.rows
        filtered_df = df.iloc[row]
        
        if len(filtered_df) > 0:
            get_job_script(_self, jobid=filtered_df.JobID.iloc[0])

    
    @st.cache_data(ttl=3600, show_spinner=False)
    def frame_group_by_user(_self, start_date, end_date, current_user, user_role, *args) -> None:

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return 
            base_query = """
                SELECT
                    eff.User,
                    COUNT(eff.JobID) AS JobCount,
                    ROUND(SUM(eff.cpu_s_reserved - eff.cpu_s_used) / 86400, 1) AS Lost_CPU_Days,
                    ROUND(SUM(eff.Elapsed * eff.NCPUS) / 86400, 1) AS cpu_days,
                    printf("%2.0f%%", 100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) AS CPUEff,
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
            params = None

            if user_role =='admin':    
                params = (start_date, end_date)
            elif user_role == 'exfel':
                base_query += "AND eff.Account = 'exfel'"
                params = (start_date, end_date) 
            elif user_role == 'user':
                base_query += " AND eff.User = ?"
                params = (start_date, end_date, current_user)

            
            df = pd.read_sql_query(base_query + " GROUP BY eff.User", _self.con, params=params)
            if df.empty:
                st.warning("No data available for the selected date range.")
                return

            if user_role == 'user':
                df = df.T.reset_index()
                df.columns = ["Metric", "Value"]
            
            st.markdown("<div style='height: 64px;'></div>", unsafe_allow_html=True)
            st.markdown("Data Grouped by User", help='Partition "jhub" and Interactive Jobs are excluded')
            st.write(df)
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def bar_char_by_user(_self, start_date, end_date, current_user, role, number, scale_efficiency, *args) -> None:
        st.markdown('Total Lost CPU-Time per User', help='Partition "jhub" and Interactive Jobs are excluded')

        # Set parameters for the SQL query
        params = (start_date, end_date)

        # Select query based on `scale_efficiency`
        if scale_efficiency:
            query = """
                SELECT 
                    eff.User,
                    COUNT(eff.JobID) AS job_count,
                    ROUND(SUM(eff.cpu_s_reserved - eff.cpu_s_used) / 86400, 1) AS lost_cpu_days,
                    (CASE 
                        WHEN (eff.CPUeff * 100) < 50 THEN (eff.CPUeff * 100) * 2
                        ELSE 100
                    END) AS adjusted_efficiency,
                    ROUND(CASE 
                        WHEN (eff.CPUeff * 100) >= 100 THEN 0
                        WHEN (eff.CPUeff * 100) = 0 THEN (SUM(eff.cpu_s_reserved - eff.cpu_s_used) / 86400) / 2
                        ELSE (SUM(eff.Elapsed * eff.NCPUS) / 2 / 86400) * ((100 - (CASE 
                            WHEN (eff.CPUeff * 100) < 50 THEN (eff.CPUeff * 100) * 2
                            ELSE 100
                        END)) / 100)
                    END, 1) AS adjusted_lost_cpu_days
                FROM eff
                JOIN slurm ON eff.JobID = slurm.JobID
                WHERE eff.Start >= ? 
                AND eff.End <= ? 
                AND eff.End IS NOT NULL 
                AND slurm.Partition != 'jhub'
                AND slurm.JobName != 'interactive'
                GROUP BY eff.User
                ORDER BY adjusted_lost_cpu_days DESC;
            """
        else:
            query = """
                SELECT 
                    eff.User,
                    ROUND(SUM((eff.cpu_s_reserved - eff.cpu_s_used) / 86400), 1) AS lost_cpu_days,
                    COUNT(eff.JobID) AS job_count,
                    ROUND(SUM(eff.Elapsed * eff.NCPUS) / 86400, 1) AS total_cpu_days
                FROM eff
                JOIN slurm ON eff.JobID = slurm.JobID
                WHERE eff.Start >= ? 
                AND eff.End <= ? 
                AND eff.End IS NOT NULL 
                AND slurm.Partition != 'jhub'
                AND slurm.JobName != 'interactive'
                GROUP BY eff.User
                ORDER BY lost_cpu_days DESC;
            """
        if role == 'exfel':
            query = query.replace("WHERE eff.Start >= ?", "WHERE eff.Start >= ? AND eff.Account = 'exfel'")
        
        result_df = pd.read_sql_query(query, _self.con, params=params)

        if scale_efficiency:
            result_df['lost_cpu_days'] = result_df['adjusted_lost_cpu_days']

        result_df = result_df.sort_values(by='lost_cpu_days', ascending=False).head(number)

        max_lost_time = result_df['lost_cpu_days'].max()
        tick_vals = np.nan_to_num(np.linspace(0, max_lost_time, num=10), nan=0)
        tick_text = [(int(val)) for val in tick_vals]
 
        fig = px.bar(result_df, x='User', y='lost_cpu_days', hover_data=['job_count'])

        fig.update_layout(
            xaxis=dict(
                title='User',
                tickangle=-45  # Rotate x-axis labels for better readability
            ),
            yaxis=dict(
                title='Total Lost CPU Time (in Days)',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text
            )
	)

        # Display the chart in Streamlit
        st.plotly_chart(fig)


    @st.cache_data(ttl=3600, show_spinner=False)
    def job_counts_by_log2(_self, start_date, end_date, current_user, user_role, number, *args) -> None:
        st.markdown('Job Count by Job Time', help='Partition "jhub" and Interactive Jobs are excluded')
        
        min_runtime = max(0, number)        
        
        df = pd.read_sql_query("""
            SELECT (End - Start) / 60 AS runtime_minutes
            FROM allocations
            WHERE Partition != 'jhub'
            AND (End - Start) / 60 >= ? AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """, _self.con, params=[number, start_date, end_date])
        

        df['runtime_minutes'] = df['runtime_minutes'].fillna(0)
        max_runtime = df['runtime_minutes'].max()
        
        bins = [0,2,5,10,20,60,120,240,480,1440, 2880,5760,11520, 23040, max_runtime]
        bins = [b for b in bins if b >= min_runtime and b <= max_runtime]  
        bins = sorted(set(bins))
        #labels = [f"{bins[i]}-{bins[i + 1]} min" for i in range(len(bins) - 1)]
        
        # Assign the runtime intervals
        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, include_lowest=True, right=False)
        df['runtime_interval'] = df['runtime_interval'].apply(timeUtils.format_interval_label)
        
        job_counts = df['runtime_interval'].value_counts().sort_index()
        job_counts_df = job_counts.reset_index()
        job_counts_df.columns = ['runtime_interval', 'job_count']

        st.write(alt.Chart(job_counts_df).mark_bar().encode(
            x=alt.X('runtime_interval:O', sort=None),
            y=alt.Y('job_count:Q', title=None),  # Entfernt den Titel der Y-Achse
        ).properties(
            width=550, 
            height=450  
        ).configure_axis(
            labelFontSize=15
        ))

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_job_runtime(_self, start_date, end_date, currrent_user, user_role, number, scale_efficiency, *args) -> None:                                       # Show Timeframe, gesamt zeit, verlorene zeit!  ;    lost_cpu_by_sate tabelle mit daten ; hyperthreading default disable takte into account

        if scale_efficiency:

            df = pd.read_sql_query("""
            SELECT
                (End - Start) / 60 AS runtime_minutes,
                ROUND(CASE 
                    WHEN (CPUeff * 100) >= 100 THEN 0
                    WHEN (CPUeff * 100) = 0 THEN (CPUTime - TotalCPU) / 86400 / 2
                    ELSE (CPUTime / 2 / 86400) * ((100 - (CASE 
                        WHEN (CPUeff * 100) < 50 THEN (CPUeff * 100) * 2
                        ELSE 100
                    END)) / 100)
                END, 1) AS lost_cpu_days,
                CPUTime 
            FROM allocations
            WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """, _self.con, params=[start_date, end_date])
            
            
            df2 = pd.read_sql_query("""
            SELECT 
                CAST(ROUND(SUM(CPUTime / 86400) / 2) AS INTEGER) AS total_cpu_days,
                CAST(ROUND(SUM(CASE 
                    WHEN (CPUeff * 100) >= 100 THEN 0
                    WHEN (CPUeff * 100) = 0 THEN (CPUTime - TotalCPU) / 86400 / 2
                    ELSE (CPUTime / 2 / 86400) * ((100 - (CASE 
                        WHEN (CPUeff * 100) < 50 THEN (CPUeff * 100) * 2
                        ELSE 100
                    END)) / 100)
                END)) AS INTEGER) AS lost_cpu_days
            FROM allocations
            WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """, _self.con, params=[start_date, end_date])
                
        else:    
            df = pd.read_sql_query("""        
            SELECT
                (End - Start) / 60 AS runtime_minutes,
                ROUND((CPUTime - TotalCPU) / 86400, 1) AS lost_cpu_days,
                CPUTime 
            FROM allocations
            WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """, _self.con, params=[start_date, end_date])

            df2 = pd.read_sql_query("""  
                    SELECT 
                    CAST(ROUND(SUM(CPUTime / 86400)) AS INTEGER) AS total_cpu_days, CAST(ROUND(SUM((CPUTime- TotalCPU) / 86400)) AS INTEGER) AS lost_cpu_days
                    FROM allocations
                    WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """, _self.con, params=[start_date, end_date])

        max_runtime = df['runtime_minutes'].max()
        bins = [0,2,5,10,20,60,120,240,480,1440, 2880,5760,11520, 23040, max_runtime]
        bins = [b for b in bins if b <= max_runtime]  # Filter by min and max
        bins = sorted(set(bins))

        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins)

        # Aggregate total CPU time by runtime interval
        cpu_time_by_interval = df.groupby('runtime_interval', observed=True)['lost_cpu_days'].sum().reset_index()

        cpu_time_by_interval['runtime_interval'] = cpu_time_by_interval['runtime_interval'].apply(
            timeUtils.format_interval_label)

        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='lost_cpu_days')
        
        st.markdown('Lost CPU Time by Job Runtime Interval',  help='Partition "jhub" and Interactive Jobs are excluded' )
        st.plotly_chart(fig)
        
        
        st.write('Cluster CPU efficiency:')
        # Calculate cluster efficiency
        df2['cluster_efficiency'] = (df2['total_cpu_days'] - df2['lost_cpu_days']) / df2['total_cpu_days'] * 100

        df2 = df2.rename(columns={
            'total_cpu_days': 'total CPU days booked',
            'lost_cpu_days': 'total CPU days lost',
            'cluster_efficiency': 'cluster efficiency (%)'
        })
        df2 = df2.transpose()
        df2.columns = ['Time in days']
        
        # Format the efficiency row
        df2.loc['cluster efficiency (%)', 'Time in days'] = f"{df2.loc['cluster efficiency (%)', 'Time in days']:.2f}%"
        
        # Convert lost and total CPU days to integers with commas
        df2.loc['total CPU days booked', 'Time in days'] = f"{int(df2.loc['total CPU days booked', 'Time in days']):,}"
        df2.loc['total CPU days lost', 'Time in days'] = f"{int(df2.loc['total CPU days lost', 'Time in days']):,}"
        
        st.dataframe(df2)



    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_batch_inter(_self, start_date, end_date, current_user, user_role, scale_efficiency, *args) -> None:

        if scale_efficiency:  
            df = pd.read_sql_query("""
                SELECT
                    ROUND(CASE 
                        WHEN (CPUeff * 100) >= 100 THEN 0
                        WHEN (CPUeff * 100) = 0 THEN (CPUTime - TotalCPU) / 86400 / 2
                        ELSE (CPUTime / 2 / 86400) * ((100 - (CASE 
                            WHEN (CPUeff * 100) < 50 THEN (CPUeff * 100) * 2
                            ELSE 100
                        END)) / 100)
                    END, 1) AS lost_cpu_days,
                    JobName, 
                    Partition 
                FROM allocations 
                WHERE Partition != 'jhub' 
                AND Start >= ? 
                AND End <= ?
                AND JobName != 'interactive'

                """, _self.con, params=[start_date, end_date])

        else:
            df = pd.read_sql_query("""
                SELECT             
                    ROUND((CPUTime - TotalCPU) / 86400, 1) AS lost_cpu_days,
                    JobName, Partition FROM allocations WHERE Partition != 'jhub' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """,_self.con, params=[start_date, end_date])

        df['category'] = df.apply(
            lambda row: 'Jupyterhub' if row['JobName'] == 'spawner-jupyterhub'
            else 'Interactive' if row['JobName'] == 'interactive'
            else 'Batch' if row['JobName'] != ''
            else 'None',
            axis=1
        )

        aggregated_df = df.groupby('category', as_index=False).agg({'lost_cpu_days': 'sum'})
        color_map = {
            'Interactive': 'red',
            'Batch': 'darkcyan',
            'None': 'grey',
            'spawner-jupyterhub': '#e377c2'
        }

        fig = px.pie(
            aggregated_df,
            names='category',
            values='lost_cpu_days',
            color='category',
            color_discrete_map=color_map
        )
        st.markdown('Lost CPU Time by Job Category', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)

    # def chart_cpu_utilization(_self) -> None:
    #     """
    #     Displays a line chart of average CPU utilization by hour from the avg_eff table.
    #     """
    #     df = pd.read_sql_query("""
    #         SELECT strftime('%Y-%m-%d %H:00:00', start) AS period, eff AS avg_efficiency
    #         FROM avg_eff
    #         GROUP BY strftime('%Y-%m-%d %H:00:00', start)
    #         ORDER BY period
    #     """, _self.con)
    #     st.line_chart(df.set_index('period'))


    @st.cache_data(ttl=3600, show_spinner=False)
    def scatter_chart_data_cpu_gpu_eff(_self, start_date, end_date, current_user, user_role, scale_efficiency, *args):
        st.markdown('CPU Efficiency by Job duration',  help='Partition "jhub" and Interactive Jobs are excluded')
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
        params = None

        if user_role == "admin":
            query += "AND Elapsed > 100 "
            params = (start_date, end_date)
        elif user_role == "exfel":
            query += "AND Account == 'exfel'"
            params = (start_date, end_date)
        elif user_role == "user":
            query += "AND User == ?"
            params = (start_date, end_date, current_user)
        
        if params is None:
            st.error("invalid user role")
            return

        df = pd.read_sql_query(query + " ORDER BY Elapsed ASC;", _self.con, params=params)
        df['GpuUtil'] = df['GpuUtil'] * 100
        
        df = df.dropna(subset=['Elapsed'])
        df['Elapsed'] = pd.to_numeric(df['Elapsed'], errors='coerce').astype(int).apply(timeUtils.seconds_to_timestring)

        # Clip CPU efficiency values to 100
        df['CPUEff'] = df['CPUEff'].clip(upper=100)

        if scale_efficiency:
            df['CPUEff'] = df['CPUEff'].where(df['CPUEff'] > 100, df['CPUEff'] * 2).clip(upper=100)

        df['lost_cpu_days'] = df['lost_cpu_days'].fillna(0)

        df['hover_text'] = df.apply(
            lambda row: f"JobID: {row['JobID']}<br>User: {row['User']}<br>Lost CPU days: {row['lost_cpu_days']}<br>CPU Efficiency: {row['CPUEff']}", axis=1
        )


        custom_color_scale = [
        [0.0, "blue"],  
        [0.25, "cyan"],
        [0.5, "green"],     
        [0.75, "orange"],  
        [1.0, "red"],       
        ]

        # Create scatter plot using Plotly Express
        fig = px.scatter(
            df, 
            x='Elapsed', 
            y='CPUEff', 
            range_color=[0,100],
            color='GpuUtil',  # Color by GpuUtil
            color_continuous_scale=custom_color_scale,  # Color scale for GpuUtil
            hover_data={'JobID': True, 'User': True, 'lost_cpu_days': True, 'CPUEff': True},  #Additional hover data
            labels={'Elapsed': 'Elapsed Time', 'CPUEff': 'CPU Efficiency (%)'}
        )
	    
        fig.update_traces(marker=dict(size=2.5))
        # Update layout to add axis titles
        fig.update_layout(
            xaxis=dict(showgrid=False),  # Disable x-axis gridlines
            yaxis=dict(showgrid=False),  # Disable y-axis gridlines
            xaxis_title='Elapsed Time',
            yaxis_title='CPU Efficiency (%)',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent figure background
        )


        # Display the plot in Streamlit
        st.plotly_chart(fig, theme=None)

    @st.cache_data(ttl=3600, show_spinner=False)
    def pie_chart_by_session_state(_self, start_date, end_date, current_user, user_role, scale_efficiency, *args):

        if scale_efficiency:
            query = f"""
                SELECT 
                    IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
                    CAST(ROUND(SUM(
                        CASE 
                            WHEN (CPUEff * 100) >= 100 THEN 0
                            WHEN (CPUEff * 100) = 0 THEN (CPUTime - TotalCPU) / 86400 / 2
                            ELSE (CPUTime / 2 / 86400) * ((100 - (CASE 
                                WHEN (CPUEff * 100) < 50 THEN (CPUEff * 100) * 2
                                ELSE 100
                            END)) / 100)
                        END
                    )) AS INTEGER) AS lost_cpu_days
                FROM allocations
                WHERE Partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' AND Start >= ? AND End <= ? AND JobName != 'interactive'
                """
        else:
            query = f"""
                SELECT 
                    IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
                    CAST(ROUND(SUM((CPUTime - TotalCPU) / 86400)) AS INTEGER) AS lost_cpu_days
                FROM allocations
                WHERE Partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' AND Start >= ? AND End <= ? AND JobName != 'interactive'
            """

        if user_role == "admin":
            params = (start_date, end_date)
        elif user_role == "exfel":
            query += " AND Account == 'exfel'"
            params = (start_date, end_date)
        elif user_role == "user":
            query += " AND User == ?"
            params = (start_date, end_date, current_user)


        df = pd.read_sql_query(query + " GROUP BY Category", _self.con, params=params)


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
    def pie_chart_by_job_count(_self,start_date, end_date, current_user, user_role, *args):

        query = """
            SELECT
                IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category, COUNT(JobID) AS JobCount
            FROM allocations
            WHERE partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' AND Start >= ? AND End <= ? AND JobName != 'interactive'
        """
        
        if user_role == 'admin':    
            query
            params = start_date, end_date
        elif user_role == 'exfel':
            query += "AND Account = 'exfel'"
            params = start_date, end_date
        elif user_role =='user':
            query += "AND User = ?"
            params=(start_date, end_date, current_user)
            
        df = pd.read_sql_query(query + " GROUP BY State", _self.con, params=params)

        fig = px.pie(
            df,
            names='Category',
            values='JobCount',
            title=f"Job Count by state",
            color='Category',
            color_discrete_map=_self.color_map
        )
        fig.update_layout(showlegend=False)
        st.markdown('Job Count by State', help='Partition "jhub" and Interactive Jobs are excluded')
        st.plotly_chart(fig)

        df_grouped = df.groupby('Category').agg({
            'JobCount': 'sum' 
        }).reset_index()

        df_grouped = df_grouped.sort_values(by='JobCount', ascending=False)

        st.dataframe(df_grouped, hide_index=True)

    # @st.cache_data(ttl=3600, show_spinner=False)
    # def efficiency_percentile_chart(_self):

    #     df = pd.read_sql_query("""
    #             SELECT CPUEff, jobID
    #             FROM allocations WHERE Partition != 'jhub'
    #         """, _self.con)

    #     df = df[df['CPUEff'] > 0]

    #     # Use qcut to create exactly 10 equal-sized bins based on cpu_efficiency
    #     df['efficiency_percentile'] = pd.qcut(df['CPUEff'], 10, labels=False)

    #     # Aggregate the data by these percentiles
    #     percentile_df = df.groupby('efficiency_percentile').agg(
    #         mean_cpu_efficiency=('CPUEff', 'mean'),
    #         min_cpu_efficiency=('CPUEff', 'min'),
    #         max_cpu_efficiency=('CPUEff', 'max'),
    #         total_jobs=('JobID', 'count')
    #     ).reset_index()

    #     # Calculate the percentage of total jobs in each percentile
    #     total_jobs = df.shape[0]
    #     percentile_df['job_percentage'] = (percentile_df['total_jobs'] / total_jobs) * 100

    #     percentile_df.columns = ['Efficiency Percentile', 'Mean Efficiency', 'Min Efficiency', 'Max Efficiency',
    #                             'Total Jobs', 'Job Percentage']

    #     # Create the figure
    #     fig = go.Figure()

    #     # Add the number of jobs as a bar trace
    #     # fig.add_trace(go.Bar(
    #     #     x=percentile_df['Efficiency Percentile'],
    #     #     y=percentile_df['Job Percentage'],  # Ensure we're referencing the correct column name
    #     #     name='Job Percentage',
    #     #     marker_color='rgba(0,100,200,0.6)'
    #     # ))

    #     # Add line trace for mean CPU efficiency
    #     fig.add_trace(go.Scatter(
    #         x=percentile_df['Efficiency Percentile'],
    #         y=percentile_df['Mean Efficiency'],
    #         mode='lines+markers',
    #         name='Mean CPU Efficiency',
    #         line=dict(color='royalblue'),
    #     ))

    #     # Add fill between the min and max efficiency for each percentile
    #     fig.add_trace(go.Scatter(
    #         x=pd.concat([percentile_df['Efficiency Percentile'], percentile_df['Efficiency Percentile'][::-1]]),
    #         y=pd.concat([percentile_df['Min Efficiency'], percentile_df['Max Efficiency'][::-1]]),
    #         fill='toself',
    #         fillcolor='rgba(0,100,80,0.2)',
    #         line=dict(color='rgba(0,100,80,0)'),
    #         name='Efficiency Range'
    #     ))

    #     # Update layout
    #     fig.update_layout(
    #         title='Distribution of Jobs and CPU Efficiency Percentiles',
    #         xaxis_title='Efficiency Percentile',
    #         yaxis_title='Percentage of Jobs / CPU Efficiency',
    #         template='plotly_white'
    #     )

    #     # Display the chart in Streamlit
    #     st.plotly_chart(fig)

