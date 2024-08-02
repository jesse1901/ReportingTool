import pyslurm
import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import requests
import hostlist
import gpu_node_data
import json
from streamlit_autorefresh import st_autorefresh
from streamlit_elements import dashboard


def timestring_to_seconds(timestring):
    if pd.isna(timestring) or timestring == '0' or timestring == 0 or timestring.strip() == '':
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
        st.write('All Data')
        df = pd.read_sql_query("""
            SELECT jobID, username, account, cpu_efficiency, lost_cpu_time, lost_cpu_time_sec, gpu_efficiency, lost_gpu_time, 
            lost_gpu_time_sec, real_time, job_cpu_time, real_time_sec, state, cores, gpu_nodes, start, end, job_name 
            FROM reportdata
            """, self.con)
        st.write(df)

    def frame_group_by_user(self) -> None:
        """
        Displays average efficiency and job count grouped by username in the Streamlit app
        """
        st.write('Data grouped by user')
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
        st.write('Total Lost CPU-Time per User')

        start_date, end_date = st.date_input(
            'Start Date und End Date',
            [datetime.today() - timedelta(days=30), datetime.today()],
        )
        end_date += timedelta(days=1)

        display_user = st.number_input(
            'Anzahl User', value=20,
        )

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

        # Sort DataFrame by total_lost_cpu_time in descending order and limit to top 20 users
        df = df.sort_values(by='total_lost_cpu_time', ascending=False).head(display_user)

        # Define constant tick values for the y-axis (vertical chart)
        max_lost_time = df['total_lost_cpu_time'].max()
        tick_vals = np.linspace(0, max_lost_time, num=10)
        tick_text = [seconds_to_timestring(int(val)) for val in tick_vals]

        # Plot vertical bar chart using Plotly
        fig = px.bar(df, x='username', y='total_lost_cpu_time')

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
    def job_counts_by_log2(self) -> None:
        st.write('Job Count by Job Time')
        df = pd.read_sql_query("""
        SELECT
            (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes
        FROM reportdata;
        """, con)
        max_runtime = df['runtime_minutes'].max()
        bins = [2 ** i for i in range(int(np.log2(max_runtime)) + 2)]
        labels = [f"{bins[i]}-{bins[i + 1]} min" for i in range(len(bins) - 1)]

        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, labels=labels, include_lowest=True)
        job_counts = df['runtime_interval'].value_counts().sort_index()
        st.bar_chart(job_counts)

    def pie_chart_job_count(self) -> None:
        st.write('Job Count by Job Time and CPU Time')

        # Query to get runtime in minutes, lost CPU time, and job CPU time
        df = pd.read_sql_query("""
        SELECT
            (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes,
            lost_cpu_time_sec,
            job_cpu_time
        FROM reportdata;
        """, con)

        # Calculate total CPU time booked
        if 'job_cpu_time' in df:
            df['job_cpu_time'] = df['job_cpu_time'].apply(timestring_to_seconds)
        else:
            df['job_cpu_time'] = 0

        df['total_cpu_time_booked'] = df['lost_cpu_time_sec'] + df['job_cpu_time']

        # Calculate bins for logarithmic intervals
        max_runtime = df['runtime_minutes'].max()
        bins = [2 ** i for i in range(int(np.log2(max_runtime)) + 2)]
        labels = [f"{bins[i]}-{bins[i + 1]} min" for i in range(len(bins) - 1)]

        # Assign each job to a runtime interval
        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, labels=labels, include_lowest=True)

        # Aggregate total CPU time by runtime interval
        cpu_time_by_interval = df.groupby('runtime_interval', observed=True)['total_cpu_time_booked'].sum().reset_index()

        # Create pie chart with Plotly
        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='total_cpu_time_booked',
                     title='Total CPU Time by Job Runtime Interval')

        st.plotly_chart(fig)
        # Plot pie chart in Streamlit

    def pie_chart_batch_inter(self) -> None:
        # Fetch data from the database
        df = pd.read_sql_query("""
            SELECT lost_cpu_time_sec, job_name FROM reportdata
        """, con)

        # Create a new column to categorize jobs, handling None and empty values
        df['category'] = df['job_name'].apply(
            lambda x: 'Interactive' if x and x.lower() == 'interactive'
            else 'Batch' if x and x.lower() != ''
            else 'None'
        )

        # Add a checkbox to allow users to hide the 'None' category
        hide_none = st.checkbox("Hide None", value=False)

        # Filter out 'None' category if the checkbox is selected
        if hide_none:
            df = df[df['category'] != 'None']

        # Aggregate the data by category
        aggregated_df = df.groupby('category', as_index=False).agg({'lost_cpu_time_sec': 'sum'})
        color_map = {
            'Interactive': 'red',
            'Batch': 'darkcyan',
            'None': 'grey'  # You can remove this if 'None' is hidden
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
            SELECT jobID, username, gpu_efficiency, cpu_efficiency, lost_cpu_time, lost_gpu_time, real_time_sec, real_time, cores, state
            FROM reportdata
            ORDER BY real_time_sec ASC;""", self.con)

        df['real_time_sec'] = pd.to_numeric(df['real_time_sec'], errors='coerce')
        df = df.dropna(subset=['real_time_sec'])
        df['real_time_sec'] = df['real_time_sec'].astype(int)
        df['real_time_sec'] = df['real_time_sec'].apply(seconds_to_timestring)

        df['lost_cpu_time'] = df['lost_cpu_time'].apply(timestring_to_seconds)
        df['log_lost_cpu_time'] = np.log1p(df['lost_cpu_time'])

        fig = px.scatter(
            df,
            x="real_time_sec",
            y="cpu_efficiency",
            color="log_lost_cpu_time",
            color_continuous_scale="blues",
            size_max=1,
            hover_data=["jobID", "username", "lost_cpu_time", "lost_gpu_time", "real_time", "cores", "state"],
            log_x=False,
            log_y=False,
            labels={"real_time_sec": "real_job_time"}
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Log Lost CPU Time"))
        fig.update_coloraxes(colorbar=dict(
            tickvals=[np.log1p(10 ** i) for i in range(0, int(np.log10(df['lost_cpu_time'].max())) + 1)],
            ticktext=[10 ** i for i in range(0, int(np.log10(df['lost_cpu_time'].max())) + 1)]
        ))
        fig.update_traces(marker=dict(size=3))

        st.plotly_chart(fig, theme=None)

    def scatter_chart_data_cpu_gpu_eff1(self):
        st.write('CPU Efficiency by Job duration')

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
        hide_gpu_none = st.checkbox("Hide GPU Jobs")

        # Ensure start_date is not after end_date
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # Load data from database with date filtering
        query = f"""
            SELECT jobID, username, gpu_efficiency, 
                   cpu_efficiency, lost_cpu_time, lost_gpu_time, real_time_sec, real_time, cores, state
            FROM reportdata
            WHERE start >= '{start_date.strftime('%Y-%m-%d')}' AND end <= '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY real_time_sec ASC;
        """
        df = pd.read_sql_query(query, self.con)

        # Data cleaning and transformation
        df['real_time_sec'] = pd.to_numeric(df['real_time_sec'], errors='coerce')
        df = df.dropna(subset=['real_time_sec'])
        df['real_time_sec'] = df['real_time_sec'].astype(int)
        df['real_time_sec'] = df['real_time_sec'].apply(seconds_to_timestring)

        # Filter dataframe based on the checkbox
        row_var = ['gpu_efficiency']
        if hide_gpu_none:
            df2 = df.dropna(subset=row_var)
            df = df.drop(df2.index)

            #df = df.dropna(subset=['gpu_efficiency'])
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

    def scatter_chart_data_cpu_gpu_eff(self):
        st.write('CPU Efficiency by Job duration')

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
        hide_gpu_none = st.checkbox("Hide GPU Jobs")
        scale_efficiency = st.checkbox("Hyperthreading")

        # Manage the button state using Streamlit session state
        if 'scale_efficiency' not in st.session_state:
            st.session_state.scale_efficiency = False


        # Update session state based on button click
        if scale_efficiency:
            st.session_state.scale_efficiency = not st.session_state.scale_efficiency

        # Ensure start_date is not after end_date
        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        # Load data from database with date filtering
        query = f"""
            SELECT jobID, username, gpu_efficiency, 
                   cpu_efficiency, lost_cpu_time, lost_gpu_time, real_time_sec, real_time, cores, state
            FROM reportdata
            WHERE start >= '{start_date.strftime('%Y-%m-%d')}' AND end <= '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY real_time_sec ASC;
        """
        df = pd.read_sql_query(query, self.con)

        # Data cleaning and transformation
        df['real_time_sec'] = pd.to_numeric(df['real_time_sec'], errors='coerce')
        df = df.dropna(subset=['real_time_sec'])
        df['real_time_sec'] = df['real_time_sec'].astype(int)
        df['real_time_sec'] = df['real_time_sec'].apply(seconds_to_timestring)

        # Filter dataframe based on the checkbox
        row_var = ['gpu_efficiency']
        if hide_gpu_none:
            df2 = df.dropna(subset=row_var)
            df = df.drop(df2.index)

        # Scale CPU efficiency if the button is toggled
        if st.session_state.scale_efficiency:
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

    def pie_chart_by_session_state(self):
        # Prüfen, ob die Gruppierung im Session-State gesetzt ist

        # SQL-Abfrage zur Aggregation der verlorenen CPU-Zeit nach der Gruppierung
        query = f"""
            SELECT state AS category, SUM(lost_cpu_time_sec) AS total_lost_cpu_time
            FROM reportdata
            GROUP BY state
        """
        df = pd.read_sql_query(query, con)

        # Erstellen des Pie-Charts mit Plotly
        fig = px.pie(
            df,
            names='category',
            values='total_lost_cpu_time',
            title=f"Lost CPU Time by state"
        )

        # Pie-Chart in Streamlit anzeigen
        st.plotly_chart(fig)
    def pie_chart_by_job_count(self):
        # Prüfen, ob die Gruppierung im Session-State gesetzt ist

        # SQL-Abfrage zur Aggregation der verlorenen CPU-Zeit nach der Gruppierung
        query = f"""
            SELECT state AS category, COUNT(jobID) AS Job_count
            FROM reportdata
            GROUP BY state
        """
        df = pd.read_sql_query(query, con)

        # Erstellen des Pie-Charts mit Plotly
        fig = px.pie(
            df,
            names='category',
            values='Job_count',
            title=f"Job Count by state"
        )

        # Pie-Chart in Streamlit anzeigen
        st.plotly_chart(fig)


    def efficiency_percentile_chart2(self):
        # Fetch the data from the database
        df = pd.read_sql_query("""
                   SELECT cpu_efficiency, jobID
                   FROM reportdata
               """, self.con)

        # Check if there are enough unique values in 'cpu_efficiency' to calculate percentiles
        if df['cpu_efficiency'].nunique() < 10:
            st.error("Nicht genügend einzigartige cpu_efficiency-Werte, um Perzentile zu berechnen.")
            return

        # Calculate percentiles for 'cpu_efficiency'
        df['efficiency_percentile'] = pd.qcut(df['cpu_efficiency'], 10, labels=False)

        # Aggregate the data by these percentiles
        percentile_df = df.groupby('efficiency_percentile').agg(
            mean_cpu_efficiency=('cpu_efficiency', 'mean'),
            median_cpu_efficiency=('cpu_efficiency', 'median'),
            min_cpu_efficiency=('cpu_efficiency', 'min'),
            max_cpu_efficiency=('cpu_efficiency', 'max'),
            std_cpu_efficiency=('cpu_efficiency', 'std')
        ).reset_index()

        # Rename columns for better readability
        percentile_df.columns = ['Efficiency Percentile', 'Mean', 'Median', 'Min', 'Max', 'Std']

        # Create a line chart using Plotly
        fig = px.line(
            percentile_df,
            x='Efficiency Percentile',
            y='Mean',
            error_y='Std',
            title='CPU Efficiency Percentile',
            labels={'Efficiency Percentile': 'Efficiency Percentile', 'Mean': 'Mean CPU Efficiency'}
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

    def efficiency_percentile_chart3(self):
        # Fetch the data from the database
        df = pd.read_sql_query("""
                   SELECT cpu_efficiency, COUNT(jobID) AS job_count
                   FROM reportdata
                   GROUP BY cpu_efficiency
               """, self.con)

        # Check if there are enough unique values in 'cpu_efficiency' to calculate percentiles
        if df['cpu_efficiency'].nunique() < 10:
            st.error("Nicht genügend einzigartige cpu_efficiency-Werte, um Perzentile zu berechnen.")
            return

        # Calculate percentiles for 'cpu_efficiency'
        df['efficiency_percentile'] = pd.qcut(df['cpu_efficiency'], 10, labels=False, duplicates='drop')

        # Aggregate the data by these percentiles
        percentile_df = df.groupby('efficiency_percentile').agg(
            mean_cpu_efficiency=('cpu_efficiency', 'mean'),
            std_cpu_efficiency=('cpu_efficiency', 'std')
        ).reset_index()

        # Rename columns for better readability
        percentile_df.columns = ['Efficiency Percentile', 'Mean', 'Std']

        # Create the figure
        fig = go.Figure()

        # Add mean line
        fig.add_trace(go.Scatter(
            x=percentile_df['Efficiency Percentile'],
            y=percentile_df['Mean'],
            mode='lines',
            name='Mean CPU Efficiency'
        ))

        # Add fill between the mean +/- std
        fig.add_trace(go.Scatter(
            x=percentile_df['Efficiency Percentile'],
            y=percentile_df['Mean'] + percentile_df['Std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=percentile_df['Efficiency Percentile'],
            y=percentile_df['Mean'] - percentile_df['Std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            name='±1 Std Dev'
        ))

        # Update layout
        fig.update_layout(
            title='CPU Efficiency Percentile with Standard Deviation',
            xaxis_title='Efficiency Percentile',
            yaxis_title='CPU Efficiency',
            template='plotly_white'
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

    def jobs_vs_efficiency_curve(self):
        # Fetch the data from the database
        df = pd.read_sql_query("""
                   SELECT cpu_efficiency, jobID
                   FROM reportdata
                   GROUP BY cpu_efficiency
               """, self.con)

        # Filter out rows where cpu_efficiency is 0
        df = df[df['cpu_efficiency'] != 0]

        # Calculate percentiles
        df['percentile'] = pd.qcut(df['cpu_efficiency'], 10, labels=False)

        # Aggregate the number of jobs by percentile
        percentile_df = df.groupby('percentile').agg(
            mean_efficiency=('cpu_efficiency', 'mean'),
            total_jobs=('jobID', 'sum')
        ).reset_index()

        # Rename columns for better readability
        percentile_df.columns = ['Percentile', 'Mean CPU Efficiency', 'Total Number of Jobs']

        # Create the curve chart using Plotly
        fig = px.line(
            percentile_df,
            x='Mean CPU Efficiency',
            y='Total Number of Jobs',
            title='Number of Jobs vs. CPU Efficiency Percentiles',
            labels={'Mean CPU Efficiency': 'Mean CPU Efficiency', 'Total Number of Jobs': 'Total Number of Jobs'}
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

if __name__ == "__main__":
    st_autorefresh(interval=60000)

    con = sqlite3.connect('reports.db')
    cur = con.cursor()
    create = CreateFigures(con)
    create.frame_user_all()
    create.frame_group_by_user()
    create.job_counts_by_log2()
    create.pie_chart_job_count()
    create.pie_chart_batch_inter()
    create.pie_chart_by_session_state()
    create.pie_chart_by_job_count()
    create.efficiency_percentile_chart2()
    create.efficiency_percentile_chart3()
    create.jobs_vs_efficiency_curve()
    # create.chart_cpu_utilization()
    create.bar_char_by_user()
    create.scatter_chart_data_cpu_gpu_eff()


   # create.scatter_chart_data_color_lost_cpu()

