import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import numpy as np
import sqlite3
import toml
from streamlit_condition_tree import condition_tree, config_from_dataframe
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from dataclasses import asdict
from config import get_config
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pyslurm


class time:
    def timestring_to_seconds(timestring):
        if pd.isna(timestring) or timestring == '0' or timestring == 0 or timestring.strip() == '':
            return 0

        if isinstance(timestring, float):
            timestring = str(int(timestring))  # Convert float to integer string

        # Ensure timestring is in string format
        timestring = str(timestring).strip()

        # Split by "T" to separate days from time
        if 'T' in timestring:
            days_part, time_part = timestring.split('T')
        else:
            days_part, time_part = '0', timestring

        # Convert days part
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
        if isinstance(total_seconds, int) and total_seconds >= 0:  # Check for integer and non-negative
            # Create a timedelta object from the total seconds
            td = timedelta(seconds=total_seconds)
            # Extract days, hours, minutes, and seconds from timedelta
            days = td.days
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            seconds = round(seconds)  # Round the seconds
            # Format the result as a string
            timestring = f"{days}T {hours}:{minutes}:{seconds}"
            return timestring
        else:
            return None

    def format_interval_label(interval):
        min_time = interval.left
        max_time = interval.right

        def format_time(minutes):
            days = int(minutes // 1440)  # 1440 minutes in a day
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


class CreateFigures:

    def __init__(_self, con):
        # Initialize the CreateFigures class with a database connection
        _self.con = sqlite3.connect('reports.db')
        _self.color_map = {
         'CANCELLED': '#1f77b4 ',    # Light Blue
         'COMPLETED': '#17becf ',    # Light Sky Blue
        'TIMEOUT': '#d62728 ',     # red
        'FAILED': '#e377c2',      # Pink
        'PREEMPTED': '#2ca02c',     # Light Green
        'NODE_FAIL': '#fcf76a'
    }
    
    def get_job_script(_self):
        jobid = st.number_input("Paste JobID:", 0)
        if jobid: 
            try:
                job = pyslurm.db.Job.load(jobid, with_script=True)
                st.code(job.script)
            except Exception as e:
                st.error(f"Error details: {e}")
    @st.cache_data
    def frame_user_all(_self, current_user, user_role) -> None:
        """
        Displays all job data.py from the reportdata table in the Streamlit app.
        """
        params = None
        st.write('All Data')
        if user_role == "admin":
            base_query = """SELECT jobID, username, account, cpu_efficiency, lost_cpu_time, 
                            gpu_efficiency, lost_gpu_time, real_time, job_cpu_time, state, 
                            gpu_nodes, start, end, job_name, partition
                            FROM reportdata ORDER BY start DESC LIMIT 100000"""
            df = pd.read_sql_query(base_query, _self.con, params=params)
            st.dataframe(df)
            
        else:
            base_query = """SELECT jobID, username, account, cpu_efficiency, lost_cpu_time, 
                            gpu_efficiency, lost_gpu_time, real_time, job_cpu_time, state, 
                            gpu_nodes, start, end, job_name, partition
                            FROM reportdata WHERE username = ?"""
            params = (current_user,)
            df = pd.read_sql_query(base_query, _self.con, params=params)
            st.dataframe(df)
    
    @st.cache_data
    def frame_group_by_user(_self, start_date, end_date, current_user, user_role) -> None:
        """
        Displays average efficiency and job count grouped by username in the Streamlit app
        """

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return  # Exit the function if there is an error in date selection

            # Define the base SQL query to retrieve data from the reportdata table
            base_query = """
                SELECT username,   

                    1 - SUM(CASE WHEN gpu_efficiency IS NULL THEN lost_cpu_time_sec ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN gpu_efficiency IS NULL THEN real_time_sec * cores ELSE 0 END), 0) AS cpu_efficiency,

                    COUNT(jobID) AS job_count,

                    1 - SUM(CASE WHEN gpu_efficiency IS NOT NULL THEN lost_gpu_time_sec ELSE 0 END) / 
                    NULLIF(SUM(CASE WHEN gpu_efficiency IS NOT NULL THEN real_time_sec * cores ELSE 0 END), 0) AS gpu_efficiency,

                    SUM(CASE WHEN gpu_efficiency IS NULL THEN lost_cpu_time_sec ELSE 0 END) AS total_lost_cpu_time,                     
                    CAST(SUM(lost_gpu_time_sec) AS INTEGER) AS total_lost_gpu_time
                FROM reportdata                
                WHERE start >= ? AND end <= ? AND partition != 'jhub'
            """

            if user_role =='admin':    
                params = (start_date, end_date)
            
            if user_role == 'user':
                base_query += " AND username = ?"
                params = (start_date, end_date, current_user)

            
            df = pd.read_sql_query(base_query + "GROUP BY username", _self.con, params=params)
            df['total_lost_gpu_time'] = pd.to_numeric(df['total_lost_gpu_time'], errors='coerce').fillna(0).astype('Int64')


            # Apply the conversion functions
            df['total_lost_cpu_time'] = df['total_lost_cpu_time'].apply(time.seconds_to_timestring)
            df['total_lost_gpu_time'] = df['total_lost_gpu_time'].apply(time.seconds_to_timestring)

            df['total_lost_gpu_time'] = df['total_lost_gpu_time'].replace("0T 0:0:0", None)

            if 'user' in st.session_state:
                df = df.T.reset_index()
                df.columns = ["Metric", "Value"]
            st.write(df)
    
    @st.cache_data
    def bar_char_by_user(_self, start_date, end_date, current_user, role, scale_efficiency, display_user) -> None:
        st.write('Total Lost CPU-Time per User')

        end_date += timedelta(days=1)

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return  # Exit if there's an error

            params = start_date, end_date

            df = pd.read_sql_query("""
                SELECT username, lost_cpu_time_sec, cpu_efficiency, cores
                FROM reportdata                
                WHERE start >= ? AND end <= ? AND partition != 'jhub'
                GROUP BY username
                ORDER BY lost_cpu_time_sec DESC
                """, _self.con, params=(start_date, end_date))

        # Convert total_lost_cpu_time to integer and format as DD T HH MM SS
            df.fillna({'lost_cpu_time_sec': 0, 'cpu_efficiency': 0, 'total_job_time': 0}, inplace=True)

            # Ensure that total_lost_cpu_time is integer and formatted correctly
            df['lost_cpu_time_sec'] = df['lost_cpu_time_sec'].astype(int)
            df['formatted_lost_cpu_time'] = df['lost_cpu_time_sec'].apply(time.seconds_to_timestring)

            # Sort DataFrame by total_lost_cpu_time in descending order and limit to top 20 users
            df = df.sort_values(by='lost_cpu_time_sec', ascending=False).head(display_user)
            
            usernames = df['username'].values
            lost_cpu_time = df['lost_cpu_time_sec'].fillna(0).values
            efficiencies = df['cpu_efficiency'].fillna(0).values
            cores = df['cores'].fillna(1).values
            
            if scale_efficiency:
                physical_cores = cores // 2

                adjusted_efficiencies = np.where(
                    efficiencies < 100,
                    (efficiencies / 50) * 100,  # Scale to 100% without hyperthreading
                    efficiencies )
                
                adjusted_lost_cpu_times = np.where(
                adjusted_efficiencies >= 100, 0,  # Set to 0 if efficiency is at 100% without hyperthreading
                lost_cpu_time * ((100 - adjusted_efficiencies) / 100)
    )
                
            result_df = pd.DataFrame({
                'username': usernames,
                'lost_cpu_time':lost_cpu_time
                    }).groupby('username').sum()

            # Define constant tick values for the y-axis (vertical chart)
            max_lost_time = result_df['lost_cpu_time'].max()
            tick_vals = np.nan_to_num(np.linspace(0, max_lost_time, num=10), nan=0)        
            tick_text = [time.seconds_to_timestring(int(val)) for val in tick_vals]


            # Plot vertical bar chart using Plotly
            fig = px.bar(result_df, x='username', y='total_lost_cpu_time')

            # Update the y-axis to display formatted time with constant tick values
            fig.update_layout(
                xaxis=dict(
                    title='Username',
                    tickangle=-45  # Rotate x-axis labels if needed for better readability
                ),
                yaxis=dict(
                    title='Total Lost CPU Time',
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text
                )
            )

            st.plotly_chart(fig)

    @st.cache_data
    def job_counts_by_log2(_self) -> None:
        st.write('Job Count by Job Time')
        df = pd.read_sql_query("""
            SELECT partition, (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes
            FROM reportdata
            WHERE partition != 'jhub'
        """, _self.con)
        max_runtime = df['runtime_minutes'].max()
        bins = [2 ** i for i in range(int(np.log2(max_runtime)) + 2)]
        labels = [f"{bins[i]}-{bins[i + 1]} min" for i in range(len(bins) - 1)]

        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, labels=labels, include_lowest=True)
        job_counts = df['runtime_interval'].value_counts().sort_index()
        st.bar_chart(job_counts)

    @st.cache_data
    def pie_chart_job_count(_self) -> None:
        # Query to get runtime in minutes, lost CPU time, and job CPU time
        df = pd.read_sql_query("""
        SELECT
            (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes,
            lost_cpu_time_sec,
            job_cpu_time
        FROM reportdata
        WHERE partition != 'jhub'
        """, _self.con)

        # Calculate total CPU time booked
        if 'job_cpu_time' in df:
            df['job_cpu_time'] = df['job_cpu_time'].apply(time.timestring_to_seconds)
        else:
            df['job_cpu_time'] = 0

        df['total_cpu_time_booked'] = df['lost_cpu_time_sec'] + df['job_cpu_time']

        # Calculate bins for logarithmic intervals
        max_runtime = df['runtime_minutes'].max()
        bins = [2 ** i for i in range(int(np.log2(max_runtime)) + 2)]

        # Assign each job to a runtime interval
        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, include_lowest=True)

        # Aggregate total CPU time by runtime interval
        cpu_time_by_interval = df.groupby('runtime_interval', observed=True)[
            'total_cpu_time_booked'].sum().reset_index()

        # Format labels to HH:MM
        cpu_time_by_interval['runtime_interval'] = cpu_time_by_interval['runtime_interval'].apply(
            time.format_interval_label)

        # Create pie chart with Plotly
        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='total_cpu_time_booked',
                     title='Total CPU Time by Job Runtime Interval')

        st.plotly_chart(fig)

    @st.cache_data
    def pie_chart_batch_inter(_self) -> None:
        # Fetch data from the database
        df = pd.read_sql_query("""
            SELECT lost_cpu_time_sec, job_name, partition FROM reportdata WHERE partition != 'jhub'
        """,_self.con)
        # Create a new column to categorize jobs, handling None and empty values
        df['category'] = df.apply(
            lambda row: 'Jupyterhub' if row['job_name'] == 'spawner-jupyterhub'
            else 'Interactive' if row['job_name'] == 'interactive'
            else 'Batch' if row['job_name'] != ''
            else 'None',
            axis=1
        )
        # Aggregate the data by category
        aggregated_df = df.groupby('category', as_index=False).agg({'lost_cpu_time_sec': 'sum'})
        color_map = {
            'Interactive': 'red',
            'Batch': 'darkcyan',
            'None': 'grey',
            'spawner-jupyterhub': '#e377c2'
            # You can remove this if 'None' is hidden
        }        # Create the pie chart
        fig = px.pie(
            aggregated_df,
            names='category',
            values='lost_cpu_time_sec',
            title="Lost CPU Time by Job Category",
            color='category',
            color_discrete_map=color_map
        )
        # Display the pie chart
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
    
    @st.cache_data
    def scatter_chart_data_cpu_gpu_eff(_self, start_date, end_date, current_user, user_role, scale_efficiency):
        st.write('CPU Efficiency by Job duration')

        # Ensure start_date is not after end_date
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # Load data from database with date filtering
        query = f"""
            SELECT jobID, username, gpu_efficiency, 
                   cpu_efficiency, lost_cpu_time, lost_gpu_time, real_time_sec, real_time, cores, state, partition
            FROM reportdata
            WHERE start >= '{start_date.strftime('%Y-%m-%d')}' AND end <= '{end_date.strftime('%Y-%m-%d')}' 
            AND partition != 'jhub' 
            ORDER BY real_time_sec ASC;
        """
        df = pd.read_sql_query(query, _self.con)

        # Data cleaning and transformation
        df['real_time_sec'] = pd.to_numeric(df['real_time_sec'], errors='coerce')
        df = df.dropna(subset=['real_time_sec'])
        df['real_time_sec'] = df['real_time_sec'].astype(int)
        df['real_time_sec'] = df['real_time_sec'].apply(time.seconds_to_timestring)

        df['cpu_efficiency'] = df['cpu_efficiency'].clip(upper=100)

        # Scale CPU efficiency if the button is toggled
        if scale_efficiency:
            # Calculate scaling factor based on cores, assuming hyperthreading
            df['cpu_efficiency'] = df.apply(
                lambda row: min(row['cpu_efficiency'] * 2, 100) if row['cpu_efficiency'] <= 100 else row[
                    'cpu_efficiency'], axis=1)

        # Create scatter plot
        fig = px.scatter(
            df,
            x="real_time_sec",
            y="cpu_efficiency",
            color="gpu_efficiency",
            color_continuous_scale="tealgrn",
            size_max=1,
            hover_data=["jobID", "username", "lost_cpu_time", "lost_gpu_time", "real_time", "cores", "state"],
            labels={"real_time_sec": "real_job_time"}
        )

        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(fig, theme=None)
    
    @st.cache_data
    def pie_chart_by_session_state(_self, current_user, user_role):
        # Prüfen, ob die Gruppierung im Session-State gesetzt ist

        # SQL-Abfrage zur Aggregation der verlorenen CPU-Zeit nach der Gruppierung
        base_query = f"""
            SELECT state AS category, SUM(lost_cpu_time_sec) AS total_lost_cpu_time
            FROM reportdata
            WHERE partition != 'jhub'
        """
        if user_role == 'admin':    
                params= None
        else:
                base_query += "AND username = ?"
                params=(current_user, )
            
        df = pd.read_sql_query(base_query + " GROUP BY state", _self.con, params=params)
        # Erstellen des Pie-Charts mit Plotly
        fig = px.pie(
            df,
            names='category',
            values='total_lost_cpu_time',
            title=f"Lost CPU Time by state",
            color='category',
            color_discrete_map=_self.color_map,
        )
        # Pie-Chart in Streamlit anzeigen
        st.plotly_chart(fig)

    def pie_chart_by_job_count(_self, current_user, user_role):
        # Prüfen, ob die Gruppierung im Session-State gesetzt ist

        # SQL-Abfrage zur Aggregation der verlorenen CPU-Zeit nach der Gruppierung
        base_query = """
            SELECT state AS category, COUNT(jobID) AS Job_count
            FROM reportdata WHERE partition != 'jhub'
        """

        if user_role == 'admin':    
                params=None
        else:
                base_query += "AND username = ?"
                params=(current_user, )
            
        df = pd.read_sql_query(base_query + " GROUP BY state", _self.con, params=params)

        # Erstellen des Pie-Charts mit Plotly
        fig = px.pie(
            df,
            names='category',
            values='Job_count',
            title=f"Job Count by state",
            color='category',
            color_discrete_map=_self.color_map
        )
        fig.update_layout(showlegend=False)
        # Pie-Chart in Streamlit anzeigen
        st.plotly_chart(fig)

    @st.cache_data
    def efficiency_percentile_chart(_self):
        # Fetch the data from the database
        df = pd.read_sql_query("""
                   SELECT cpu_efficiency, jobID
                   FROM reportdata WHERE partition != 'jhub'
               """, _self.con)

        # Filter out rows where cpu_efficiency is 0
        df = df[df['cpu_efficiency'] != 0]

        # Calculate percentiles for 'cpu_efficiency'
        df['efficiency_percentile'] = pd.qcut(df['cpu_efficiency'], 10, labels=False, duplicates='drop')

        # Aggregate the data by these percentiles
        percentile_df = df.groupby('efficiency_percentile').agg(
            mean_cpu_efficiency=('cpu_efficiency', 'mean'),
            min_cpu_efficiency=('cpu_efficiency', 'min'),
            max_cpu_efficiency=('cpu_efficiency', 'max'),
            total_jobs=('jobID', 'count')
        ).reset_index()

        # Calculate the percentage of total jobs in each percentile
        total_jobs = percentile_df['total_jobs'].sum()
        percentile_df['job_percentage'] = (percentile_df['total_jobs'] / total_jobs) * 100

        # Rename columns for better readability
        percentile_df.columns = ['Efficiency Percentile', 'Mean Efficiency', 'Min Efficiency', 'Max Efficiency',
                                 'Total Jobs', 'Job Percentage']

        # Create the figure
        fig = go.Figure()

        # Add the number of jobs as a bar trace
        fig.add_trace(go.Bar(
            x=percentile_df['Efficiency Percentile'],
            y=percentile_df['Job Percentage'],
            name='Job Percentage',
            marker_color='rgba(0,100,200,0.6)'
        ))

        # Add line trace for mean CPU efficiency
        fig.add_trace(go.Scatter(
            x=percentile_df['Efficiency Percentile'],
            y=percentile_df['Mean Efficiency'],
            mode='lines+markers',
            name='Mean CPU Efficiency',
            line=dict(color='royalblue'),
        ))

        # Add fill between the min and max efficiency for each percentile
        fig.add_trace(go.Scatter(
            x=pd.concat([percentile_df['Efficiency Percentile'], percentile_df['Efficiency Percentile'][::-1]]),
            y=pd.concat([percentile_df['Min Efficiency'], percentile_df['Max Efficiency'][::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(0,100,80,0)'),
            name='Efficiency Range'
        ))

        # Update layout
        fig.update_layout(
            title='Distribution of Jobs and CPU Efficiency Percentiles',
            xaxis_title='Efficiency Percentile',
            yaxis_title='Percentage / CPU Efficiency / Std Dev',
            template='plotly_white'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)