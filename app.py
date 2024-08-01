import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
from datetime import timedelta, datetime
from streamlit_autorefresh import st_autorefresh


# Utility Functions
def timestring_to_seconds(timestring):
    if pd.isna(timestring) or timestring in ['0', 0, '']:
        return 0

    if isinstance(timestring, float):
        timestring = str(int(timestring))

    timestring = str(timestring).strip()
    days, time_part = (timestring.split('T') + ['0'])[:2]

    days = int(days.strip()) if days.strip() else 0
    time_parts = time_part.split(':')
    hours = int(time_parts[0].strip()) if len(time_parts) > 0 and time_parts[0].strip() else 0
    minutes = int(time_parts[1].strip()) if len(time_parts) > 1 and time_parts[1].strip() else 0
    seconds = int(time_parts[2].strip()) if len(time_parts) > 2 and time_parts[2].strip() else 0

    total_seconds = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
    return total_seconds


def seconds_to_timestring(total_seconds):
    td = timedelta(seconds=total_seconds)
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    timestring = f"{days}T {hours:02}:{minutes:02}:{seconds:02}"
    return timestring


# Dashboard Class
class Dashboard:
    def __init__(self, con):
        self.con = con

    def frame_user_all(self):
        st.write('### All Job Data')
        query = """
        SELECT jobID, username, account, cpu_efficiency, lost_cpu_time, lost_cpu_time_sec, gpu_efficiency, lost_gpu_time, 
        lost_gpu_time_sec, real_time, job_cpu_time, real_time_sec, state, cores, gpu_nodes, start, end, job_name 
        FROM reportdata
        """
        df = pd.read_sql_query(query, self.con)
        st.dataframe(df)

    def frame_group_by_user(self):
        st.write('### Data Grouped by User')
        start_date, end_date = st.date_input(
            'Select Date Range',
            [datetime.today() - timedelta(days=30), datetime.today()],
        )
        end_date += timedelta(days=1)

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return

        query = f"""
        SELECT username, 
            AVG(IFNULL(cpu_efficiency, 0)) AS avg_cpu_efficiency, 
            AVG(IFNULL(gpu_efficiency, 0)) AS avg_gpu_efficiency,
            COUNT(jobID) AS anzahl_jobs, 
            SUM(IFNULL(lost_cpu_time_sec, 0)) AS total_lost_cpu_time, 
            SUM(IFNULL(lost_gpu_time_sec, 0)) AS total_lost_gpu_time
        FROM reportdata
        WHERE start >= '{start_date}' AND end <= '{end_date}'
        GROUP BY username
        """
        df = pd.read_sql_query(query, self.con)
        df['total_lost_cpu_time'] = df['total_lost_cpu_time'].apply(seconds_to_timestring)
        df['total_lost_gpu_time'] = df['total_lost_gpu_time'].apply(seconds_to_timestring)
        st.dataframe(df)

    def bar_chart_by_user(self):
        st.write('### Total Lost CPU-Time per User')
        start_date, end_date = st.date_input(
            'Select Date Range',
            [datetime.today() - timedelta(days=30), datetime.today()],
        )
        end_date += timedelta(days=1)
        display_user = st.number_input('Number of Top Users', value=20)

        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
                return

        query = f"""
        SELECT username, 
            AVG(IFNULL(cpu_efficiency, 0)) AS avg_cpu_efficiency, 
            AVG(IFNULL(gpu_efficiency, 0)) AS avg_gpu_efficiency,
            COUNT(jobID) AS anzahl_jobs, 
            SUM(IFNULL(lost_cpu_time_sec, 0)) AS total_lost_cpu_time
        FROM reportdata
        WHERE start >= '{start_date}' AND end <= '{end_date}'
        GROUP BY username
        ORDER BY total_lost_cpu_time DESC
        """
        df = pd.read_sql_query(query, self.con)
        df['total_lost_cpu_time'] = df['total_lost_cpu_time'].apply(seconds_to_timestring)
        df = df.head(display_user)

        fig = px.bar(df, x='username', y='total_lost_cpu_time', title='Total Lost CPU-Time per User')
        st.plotly_chart(fig)

    def pie_chart_job_count(self):
        st.write('### Job Count by Job Time and CPU Time')
        query = """
        SELECT
            (julianday(end) - julianday(start)) * 24 * 60 AS runtime_minutes,
            lost_cpu_time_sec,
            job_cpu_time
        FROM reportdata
        """
        df = pd.read_sql_query(query, self.con)
        df['job_cpu_time'] = df['job_cpu_time'].apply(timestring_to_seconds)
        df['total_cpu_time_booked'] = df['lost_cpu_time_sec'] + df['job_cpu_time']

        max_runtime = df['runtime_minutes'].max()
        bins = [2 ** i for i in range(int(np.log2(max_runtime)) + 2)]
        labels = [f"{bins[i]}-{bins[i + 1]} min" for i in range(len(bins) - 1)]
        df['runtime_interval'] = pd.cut(df['runtime_minutes'], bins=bins, labels=labels, include_lowest=True)

        cpu_time_by_interval = df.groupby('runtime_interval', observed=True)[
            'total_cpu_time_booked'].sum().reset_index()
        fig = px.pie(cpu_time_by_interval, names='runtime_interval', values='total_cpu_time_booked',
                     title='Total CPU Time by Job Runtime Interval')
        st.plotly_chart(fig)

    def pie_chart_batch_inter(self):
        st.write('### Lost CPU Time by Job Category')
        query = "SELECT lost_cpu_time_sec, job_name FROM reportdata"
        df = pd.read_sql_query(query, self.con)
        df['category'] = df['job_name'].apply(
            lambda x: 'Interactive' if x and x.lower() == 'interactive'
            else 'Batch' if x and x.lower() != ''
            else 'None'
        )
        hide_none = st.checkbox("Hide 'None' Category", value=False)
        if hide_none:
            df = df[df['category'] != 'None']
        aggregated_df = df.groupby('category', as_index=False).agg({'lost_cpu_time_sec': 'sum'})
        color_map = {'Interactive': 'red', 'Batch': 'darkcyan', 'None': 'grey'}
        fig = px.pie(aggregated_df, names='category', values='lost_cpu_time_sec', title="Lost CPU Time by Job Category",
                     color='category', color_discrete_map=color_map)
        st.plotly_chart(fig)

    def scatter_chart_data_cpu_gpu_eff(self):
        st.write('### CPU Efficiency by Job Duration')
        date_query = "SELECT MIN(start) AS min_date, MAX(end) AS max_date FROM reportdata"
        date_range_df = pd.read_sql_query(date_query, self.con)
        min_date = pd.to_datetime(date_range_df['min_date'].values[0]).date()
        max_date = pd.to_datetime(date_range_df['max_date'].values[0]).date()
        max_date += timedelta(days=1)

        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        hide_gpu_none = st.checkbox("Hide GPU Jobs")

        if start_date > end_date:
            st.error("Error: End date must be after start date.")
            return

        query = f"""
        SELECT jobID, username, gpu_efficiency, cpu_efficiency, lost_cpu_time, lost_gpu_time, real_time_sec, real_time, cores, state
        FROM reportdata
        WHERE start >= '{start_date.strftime('%Y-%m-%d')}' AND end <= '{end_date.strftime('%Y-%m-%d')}'
        ORDER BY real_time_sec
        """
        df = pd.read_sql_query(query, self.con)
        if hide_gpu_none:
            df = df[df['gpu_efficiency'] > 0]
        fig = px.scatter(df, x='real_time_sec', y='cpu_efficiency', color='gpu_efficiency',
                         title='CPU Efficiency by Job Duration',
                         labels={"real_time_sec": "Job Duration (s)", "cpu_efficiency": "CPU Efficiency (%)"})
        st.plotly_chart(fig)


def main():
    st.set_page_config(page_title="Cluster Dashboard", layout="wide")
    st_autorefresh(interval=60 * 60)  # Auto-refresh every hour

    conn = sqlite3.connect("cluster_data.db")
    dashboard = Dashboard(conn)

    st.title("Cluster Data Dashboard")

    menu = ["All Data", "Group by User", "Lost CPU Time", "Lost CPU Time by Category", "CPU Efficiency by Duration"]
    choice = st.sidebar.selectbox("Select a Section", menu)

    if choice == "All Data":
        dashboard.frame_user_all()
    elif choice == "Group by User":
        dashboard.frame_group_by_user()
    elif choice == "Lost CPU Time":
        dashboard.bar_chart_by_user()
    elif choice == "Lost CPU Time by Category":
        dashboard.pie_chart_batch_inter()
    elif choice == "CPU Efficiency by Duration":
        dashboard.scatter_chart_data_cpu_gpu_eff()


if __name__ == "__main__":
    main()
