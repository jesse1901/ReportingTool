import pyslurm
import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import numpy as np
import sqlite3
import plotly.express as px
import requests
import hostlist
import gpu_node_data
import json
from streamlit_autorefresh import st_autorefresh

def timestring_to_seconds(timestring):
    if pd.isna(timestring) or timestring == '0' or timestring == 0:
        return 0

    if isinstance(timestring, float):
        timestring = str(int(timestring))  # Convert float to integer string

    # Ensure timestring is in string format
    timestring = str(timestring).strip()

    # Split by 'T' to separate days from time
    if 'T' in timestring:
        days_part, time_part = timestring.split('T')
    else:
        days_part, time_part = '0', timestring

    # Convert days part
    days = int(days_part.strip()) if days_part.strip() else 0

    # Convert time part (HH:MM:SS)
    time_parts = time_part.split(':')
    hours = int(time_parts[0].strip()) if len(time_parts) > 0 else 0
    minutes = int(time_parts[1].strip()) if len(time_parts) > 1 else 0
    seconds = int(time_parts[2].strip()) if len(time_parts) > 2 else 0

    # Calculate total seconds
    total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds


def seconds_to_timestring(total_seconds):
    # Create a timedelta object from the total seconds
    td = timedelta(seconds=total_seconds)
    # Extract days, hours, minutes, and seconds from timedelta
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    seconds = round(seconds)
    # Format the result as a string
    timestring = f"{days}T {hours}:{minutes}:{seconds}"
    return timestring


class CreateFigures:
    def __init__(self, con):
        # Initialize the CreateFigures class with a database connection
        self.con = con

    def frame_user_all(self) -> None:
        """
        Displays all job data.py from the reportdata table in the Streamlit app.
        """
        df = pd.read_sql_query("""
            SELECT jobID, username, account, cpu_efficiency, lost_cpu_time, lost_cpu_time_sec, gpu_efficiency, lost_gpu_time, 
            lost_gpu_time_sec, real_time, job_cpu_time, job_cpu_time_s AS realtime_in_s, state, cores, gpu_nodes, start, end 
            FROM reportdata
            """, self.con)
        st.write(df)

    def frame_group_by_user(self) -> None:
        """
        Displays average efficiency and job count grouped by username in the Streamlit app
        """

        # Get start and end dates from Streamlit date input
        start_date, end_date = st.date_input(
            'Start Date - End Date',
            [datetime.today() - timedelta(days=30), datetime.today()],
        )
        end_date += timedelta(days=1)

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return  # Exit if there's an error
        df = pd.read_sql_query(f"""
                        SELECT username, 
                           AVG(IFNULL(cpu_efficiency, 0)) AS avg_cpu_efficiency, 
                           AVG(IFNULL(gpu_efficiency, 0)) AS avg_gpu_efficiency,
                           COUNT(jobID) AS anzahl_jobs, 
                           SUM(IFNULL(lost_cpu_time_sec, 0)) AS total_lost_cpu_time, 
                           SUM(IFNULL(lost_gpu_time_sec, 0)) AS total_lost_gpu_time
                        FROM reportdata
                        WHERE start >= '{start_date}' AND end <= '{end_date}'
                        GROUP BY username
        """, con)

        df['total_lost_cpu_time'] = pd.to_numeric(df['total_lost_cpu_time'], errors='coerce')
        df['total_lost_gpu_time'] = pd.to_numeric(df['total_lost_gpu_time'], errors='coerce')

        # Drop rows where the conversion failed
        df = df.dropna(subset=['total_lost_cpu_time', 'total_lost_gpu_time'])

        # Convert the columns to integers
        df['total_lost_cpu_time'] = df['total_lost_cpu_time'].astype(int)
        df['total_lost_gpu_time'] = df['total_lost_gpu_time'].astype(int)

        # Apply the conversion functions
        df['total_lost_cpu_time'] = df['total_lost_cpu_time'].apply(seconds_to_timestring)
        df['total_lost_gpu_time'] = df['total_lost_gpu_time'].apply(seconds_to_timestring)
        st.write(df)

    def bar_char_by_user(self) -> None:

        start_date, end_date = st.date_input(
            'Start Date und End Date',
            [datetime.today() - timedelta(days=30), datetime.today()],
        )
        end_date += timedelta(days=1)

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return  # Exit if there's an error
        df = pd.read_sql_query(f"""
                        SELECT username, 
                           AVG(IFNULL(cpu_efficiency, 0)) AS avg_cpu_efficiency, 
                           AVG(IFNULL(gpu_efficiency, 0)) AS avg_gpu_efficiency,
                           COUNT(jobID) AS anzahl_jobs, 
                           SUM(IFNULL(lost_cpu_time_sec, 0)) AS total_lost_cpu_time, 
                           SUM(IFNULL(lost_gpu_time_sec, 0)) AS total_lost_gpu_time
                        FROM reportdata
                        WHERE start >= '{start_date}' AND end <= '{end_date}'
                        GROUP BY username
                        ORDER BY lost_cpu_time_sec DESC
        """, con)
        # Convert total_lost_cpu_time to integer and format as DD T HH MM SS
        df['total_lost_cpu_time'] = df['total_lost_cpu_time'].astype(int)
        df['formatted_lost_cpu_time'] = df['total_lost_cpu_time'].apply(seconds_to_timestring)

        # Sort DataFrame by total_lost_cpu_time in descending order
        df = df.sort_values(by='total_lost_cpu_time', ascending=False)

        # Define constant tick values for the y-axis
        max_lost_time = df['total_lost_cpu_time'].max()
        tick_vals = np.linspace(0, max_lost_time, num=10)
        tick_text = [seconds_to_timestring(int(val)) for val in tick_vals]

        fig = px.bar(df, x='username', y='total_lost_cpu_time', text='formatted_lost_cpu_time',
                     title='Total Lost CPU Time by User')

        # Update y-axis to show constant tick values with formatted time
        fig.update_traces(textposition='inside')
        fig.update_layout(
            yaxis=dict(
                title='Total Lost CPU Time',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text
            )
        )

        st.plotly_chart(fig)

    def chart_cpu_utilization(self) -> None:
        """
        Displays a line chart of average CPU utilization by hour from the avg_eff table.
        """
        df = pd.read_sql_query("""
            SELECT strftime('%Y-%m-%d %H:00:00', start) AS period, eff AS avg_efficiency
            FROM avg_eff
            GROUP BY strftime('%Y-%m-%d %H:00:00', start)
            ORDER BY period
        """, self.con)
        st.line_chart(df.set_index('period'))

    def scatter_chart_data_color_lost_cpu(self):
        df = pd.read_sql_query("""
            SELECT jobID, username, gpu_efficiency, cpu_efficiency, lost_cpu_time, lost_gpu_time, job_cpu_time_s, real_time, cores, state
            FROM reportdata
            ORDER BY job_cpu_time_s ASC;""", self.con)

        df['job_cpu_time_s'] = pd.to_numeric(df['job_cpu_time_s'], errors='coerce')
        df = df.dropna(subset=['job_cpu_time_s'])
        df['job_cpu_time_s'] = df['job_cpu_time_s'].astype(int)
        df['job_cpu_time_s'] = df['job_cpu_time_s'].apply(seconds_to_timestring)

        df['lost_cpu_time'] = df['lost_cpu_time'].apply(timestring_to_seconds)
        df['log_lost_cpu_time'] = np.log1p(df['lost_cpu_time'])

        fig = px.scatter(
            df,
            x="job_cpu_time_s",
            y="cpu_efficiency",
            color="log_lost_cpu_time",
            color_continuous_scale="blues",
            size_max=1,
            hover_data=["jobID", "username", "lost_cpu_time", "lost_gpu_time", "real_time", "cores", "state"],
            log_x=False,
            log_y=False,
            labels={"job_cpu_time_s": "real_job_time"}
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Log Lost CPU Time"))
        fig.update_coloraxes(colorbar=dict(
            tickvals=[np.log1p(10 ** i) for i in range(0, int(np.log10(df['lost_cpu_time'].max())) + 1)],
            ticktext=[10 ** i for i in range(0, int(np.log10(df['lost_cpu_time'].max())) + 1)]
        ))
        fig.update_traces(marker=dict(size=3))

        st.plotly_chart(fig, theme=None)

    def scatter_chart_data_cpu_gpu_eff(self):
        # Fetch the available date range from the database
        date_query = """
            SELECT MIN(start) AS min_date, MAX(end) AS max_date
            FROM reportdata
        """
        date_range_df = pd.read_sql_query(date_query, self.con)
        min_date = pd.to_datetime(date_range_df['min_date'].values[0]).date()
        max_date = pd.to_datetime(date_range_df['max_date'].values[0]).date()
        max_date += timedelta(days=1)

        # Create a slider for date range selection
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )

        # Ensure start_date is not after end_date
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # Load data from database with date filtering
        query = f"""
            SELECT jobID, username, gpu_efficiency, 
                   cpu_efficiency, lost_cpu_time, lost_gpu_time, job_cpu_time_s, real_time, cores, state
            FROM reportdata
            WHERE start >= '{start_date.strftime('%Y-%m-%d')}' AND end <= '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY job_cpu_time_s ASC;
        """
        df = pd.read_sql_query(query, self.con)

        # Data cleaning and transformation
        df['job_cpu_time_s'] = pd.to_numeric(df['job_cpu_time_s'], errors='coerce')
        df = df.dropna(subset=['job_cpu_time_s'])
        df['job_cpu_time_s'] = df['job_cpu_time_s'].astype(int)
        df['job_cpu_time_s'] = df['job_cpu_time_s'].apply(seconds_to_timestring)

        hide_gpu_none = st.checkbox("Hide GPU Jobs")

        # Filter dataframe based on the checkbox
        row_var = ['gpu_efficiency']
        if hide_gpu_none:
            df = df[~df[row_var].isnull()]
            #df = df.dropna(subset=['gpu_efficiency'])
        # Create scatter plot
        fig = px.scatter(
            df,
            x="job_cpu_time_s",
            y="cpu_efficiency",
            color="gpu_efficiency",
            color_continuous_scale="tealgrn",
            size_max=1,
            hover_data=["jobID", "username", "lost_cpu_time", "lost_gpu_time", "real_time", "cores", "state"],
            labels={"job_cpu_time_s": "real_job_time"}
        )

        fig.update_traces(marker=dict(size=3))
        st.plotly_chart(fig, theme=None)

if __name__ == "__main__":
    st_autorefresh(interval=10000)
    con = sqlite3.connect('reports.db')
    cur = con.cursor()
    create = CreateFigures(con)
    create.frame_user_all()
    create.frame_group_by_user()
    # create.chart_cpu_utilization()
    create.bar_char_by_user()
    create.scatter_chart_data_cpu_gpu_eff()
    create.scatter_chart_data_color_lost_cpu()

