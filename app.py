import pyslurm
import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from streamlit_condition_tree import condition_tree, config_from_dataframe
from streamlit_keycloak import login
from dataclasses import asdict

color_map = {
    'CANCELLED': '#1f77b4 ',    # Light Blue
    'COMPLETED': '#17becf ',    # Light Sky Blue
    'TIMEOUT': '#d62728 ',     # red
    'FAILED': '#e377c2',      # Pink
    'PREEMPTED': '#2ca02c',     # Light Green
    'NODE_FAIL': '#fcf76a'
}


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
    def __init__(self, con):
        # Initialize the CreateFigures class with a database connection
        self.con = con

    def frame_user_all(self) -> None:
        """
        Displays all job data.py from the reportdata table in the Streamlit app.
        """
        # st.write('All Data')
        # sql_select = ["jobID", "username", "account", "cpu_efficiency", "lost_cpu_time", "lost_cpu_time_sec", "gpu_efficiency", "lost_gpu_time",
        #     "lost_gpu_time_sec", "real_time", "job_cpu_time", "real_time_sec", "state", "cores", "gpu_nodes", "start", "end", "job_name", "partition" ]
        # select = st.multiselect('Select Data', sql_select)
        #
        # sql_where_condition = st.text_input("WHERE", value="WHERE")
        #
        # if st.button("Abfrage ausführen"):
        #     # Erstelle die SQL-Abfrage
        #     selected_columns_str = ", ".join(sql_select)
        #
        #     # Stelle sicher, dass die WHERE-Klausel gültig ist
        #     if sql_where_condition.strip() == "WHERE" or sql_where_condition.strip() == "":
        #         sql_query = f'SELECT {selected_columns_str} FROM reportdata'
        #     else:
        #         sql_query = f'SELECT {selected_columns_str} FROM reportdata {sql_where_condition}'

            # Zeige die SQL-Abfrage in Streamlit an
        sql_query = "SELECT * FROM reportdata"
        df = pd.read_sql_query(sql_query, self.con)
        config = config_from_dataframe(df)
        query_string = condition_tree(config)

        df = df.query(query_string)
        st.dataframe(df)
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
                           SUM(IFNULL(lost_gpu_time_sec, 0)) AS total_lost_gpu_time,
                           partition
                        FROM reportdata
                        WHERE start >= '{start_date}' AND end <= '{end_date}' AND partition != 'jhub'
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
                           SUM(IFNULL(lost_gpu_time_sec, 0)) AS total_lost_gpu_time,
                           partition
                        FROM reportdata
                        WHERE start >= '{start_date}' AND end <= '{end_date}' AND partition != 'jhub'  AND gpu_efficiency IS NULL
                        GROUP BY username
                        ORDER BY lost_cpu_time_sec DESC
        """, con)
        # Convert total_lost_cpu_time to integer and format as DD T HH MM SS
        df.fillna({'total_lost_cpu_time': 0, 'avg_cpu_efficiency': 0, 'total_job_time': 0}, inplace=True)

        # Ensure that total_lost_cpu_time is integer and formatted correctly
        df['total_lost_cpu_time'] = df['total_lost_cpu_time'].astype(int)
        df['formatted_lost_cpu_time'] = df['total_lost_cpu_time'].apply(seconds_to_timestring)

        # Sort DataFrame by total_lost_cpu_time in descending order and limit to top 20 users
        df = df.sort_values(by='total_lost_cpu_time', ascending=False).head(display_user)

        scale_efficiency = st.checkbox("Hyperthreading Aus")
        if scale_efficiency:
            df['total_job_time'] = np.where(df['avg_cpu_efficiency'] != 100, df['total_lost_cpu_time'] / ((100 - df['avg_cpu_efficiency']) / 100), np.nan)
            df['cpu_efficiency'] = df.apply(
                lambda row: min(row['avg_cpu_efficiency'] * 2, 100) if row['avg_cpu_efficiency'] <= 100 else row['avg_cpu_efficiency'], axis=1)

            # Handle NaN in cpu_efficiency and total_job_time
            df['cpu_efficiency'] = df['cpu_efficiency'].fillna(0)
            df['total_lost_cpu_time'] = np.where(df['cpu_efficiency'] != 0, df['total_job_time'] / df['cpu_efficiency'], np.nan)
            df['total_lost_cpu_time'] = df['total_lost_cpu_time'].fillna(0)

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
            SELECT partition, (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes
            FROM reportdata
            WHERE partition != 'jhub'
        """, con)
        max_runtime = df['runtime_minutes'].max()
        bins = [2 ** i for i in range(int(np.log2(max_runtime)) + 2)]
        labels = [f"{bins[i]}-{bins[i + 1]} min" for i in range(len(bins) - 1)]

        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, labels=labels, include_lowest=True)
        job_counts = df['runtime_interval'].value_counts().sort_index()
        st.bar_chart(job_counts)

    def pie_chart_job_count(self) -> None:
        # Query to get runtime in minutes, lost CPU time, and job CPU time
        df = pd.read_sql_query("""
        SELECT
            (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes,
            lost_cpu_time_sec,
            job_cpu_time
        FROM reportdata
        WHERE partition != 'jhub'
        """, self.con)

        # Calculate total CPU time booked
        if 'job_cpu_time' in df:
            df['job_cpu_time'] = df['job_cpu_time'].apply(timestring_to_seconds)
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
            format_interval_label)

        # Create pie chart with Plotly
        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='total_cpu_time_booked',
                     title='Total CPU Time by Job Runtime Interval')

        st.plotly_chart(fig)

    def pie_chart_batch_inter(self) -> None:
        # Fetch data from the database
        df = pd.read_sql_query("""
            SELECT lost_cpu_time_sec, job_name, partition FROM reportdata WHERE partition != 'jhub'
        """, con)
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

    # def chart_cpu_utilization(self) -> None:
    #     """
    #     Displays a line chart of average CPU utilization by hour from the avg_eff table.
    #     """
    #     df = pd.read_sql_query("""
    #         SELECT strftime('%Y-%m-%d %H:00:00', start) AS period, eff AS avg_efficiency
    #         FROM avg_eff
    #         GROUP BY strftime('%Y-%m-%d %H:00:00', start)
    #         ORDER BY period
    #     """, self.con)
    #     st.line_chart(df.set_index('period'))

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
                   cpu_efficiency, lost_cpu_time, lost_gpu_time, real_time_sec, real_time, cores, state, partition
            FROM reportdata
            WHERE start >= '{start_date.strftime('%Y-%m-%d')}' AND end <= '{end_date.strftime('%Y-%m-%d')}' 
            AND partition != 'jhub' 
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
            WHERE partition != 'jhub'
            GROUP BY state
        """
        df = pd.read_sql_query(query, con)

        # Erstellen des Pie-Charts mit Plotly
        fig = px.pie(
            df,
            names='category',
            values='total_lost_cpu_time',
            title=f"Lost CPU Time by state",
            color='category',
            color_discrete_map=color_map,
        )
        # Pie-Chart in Streamlit anzeigen
        st.plotly_chart(fig)

    def pie_chart_by_job_count(self):
        # Prüfen, ob die Gruppierung im Session-State gesetzt ist

        # SQL-Abfrage zur Aggregation der verlorenen CPU-Zeit nach der Gruppierung
        query = f"""
            SELECT state AS category, COUNT(jobID) AS Job_count
            FROM reportdata WHERE partition != 'jhub'
            GROUP BY state
        """
        df = pd.read_sql_query(query, con)

        # Erstellen des Pie-Charts mit Plotly
        fig = px.pie(
            df,
            names='category',
            values='Job_count',
            title=f"Job Count by state",
            color='category',
            color_discrete_map=color_map
        )
        fig.update_layout(showlegend=False)
        # Pie-Chart in Streamlit anzeigen
        st.plotly_chart(fig)

    def efficiency_percentile_chart(self):
        # Fetch the data from the database
        df = pd.read_sql_query("""
                   SELECT cpu_efficiency, jobID
                   FROM reportdata WHERE partition != 'jhub'
               """, self.con)

        # Filter out rows where cpu_efficiency is 0
        df = df[df['cpu_efficiency'] != 0]

        # Calculate percentiles for 'cpu_efficiency'
        df['efficiency_percentile'] = pd.qcut(df['cpu_efficiency'], 10, labels=False, duplicates='drop')

        # Aggregate the data by these percentiles
        percentile_df = df.groupby('efficiency_percentile').agg(
            mean_cpu_efficiency=('cpu_efficiency', 'mean'),
            min_cpu_efficiency=('cpu_efficiency', 'min'),
            max_cpu_efficiency=('cpu_efficiency', 'max'),
            std_cpu_efficiency=('cpu_efficiency', 'std'),
            total_jobs=('jobID', 'count')
        ).reset_index()

        # Calculate the percentage of total jobs in each percentile
        total_jobs = percentile_df['total_jobs'].sum()
        percentile_df['job_percentage'] = (percentile_df['total_jobs'] / total_jobs) * 100

        # Rename columns for better readability
        percentile_df.columns = ['Efficiency Percentile', 'Mean Efficiency', 'Min Efficiency', 'Max Efficiency',
                                 'Std Dev Efficiency', 'Total Jobs', 'Job Percentage']

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
            line=dict(color='royalblue')
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

        # Add line trace for standard deviation
        fig.add_trace(go.Scatter(
            x=percentile_df['Efficiency Percentile'],
            y=percentile_df['Std Dev Efficiency'],
            mode='lines',
            name='Std Dev Efficiency',
            line=dict(color='orange', dash='dash')
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


def main():
    # Tabs erstellen
    tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data", "Efficiency", "Total"])

    with tab1:
        st.header("User Data")
        col1, col2 = st.columns([3, 1])
        with col1:
            create.frame_user_all()
        with col2:
            create.frame_group_by_user()

    with tab2:
        st.header("Job Data")
        col3, col4, col5 = st.columns(3)
        with col3:
            create.job_counts_by_log2()
        with col4:
            create.pie_chart_job_count()
        with col5:
            create.pie_chart_batch_inter()

    with tab3:
        st.header("Efficiency")
        col6, col7, col8 = st.columns(3)
        with col6:
            create.pie_chart_by_session_state()
        with col7:
            create.pie_chart_by_job_count()
        with col8:
            create.efficiency_percentile_chart()
            # create.chart_cpu_utilization()

    with tab4:
        st.header("")
        col9, col10 = st.columns(2)
        with col9:
            create.bar_char_by_user()
        with col10:
            create.scatter_chart_data_cpu_gpu_eff()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    con = sqlite3.connect('reports.db')
    create = CreateFigures(con)
    st_autorefresh(interval=600000)
    main()

    #     st.title("Streamlit Keycloak example")
    #
    #     key = login(
    #         url="https://keycloak.desy.de/auth",
    #         realm="testing",
    #         client_id="maxwell-reporting-oidc",
    #         init_options={
    #             "pkceMethod": "S256",
    #             "checkLoginIframe": False
    #         }
    #     )
    #     print(key)
    #     if key.authenticated:
    #         st.write('Authenticated')



