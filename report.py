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


def count_keys_under_steps(d):
    """
    Extracts keys under the 'steps' key from a dictionary.

    Args:
        d (dict): Dictionary containing 'steps' key.

    Returns:
        list: List of keys under 'steps', or an empty list if 'steps' is not a dictionary.
    """
    steps_dict = d.get('steps', {})
    if isinstance(steps_dict, dict):
        return list(steps_dict.keys())
    return []


def timestring_to_seconds(timestring):
    parts = timestring.split('T')
    days = int(parts[0])
    hms = parts[1].split(':')
    hours = int(hms[0])
    minutes = int(hms[1])
    seconds = int(hms[2])
    round(seconds)
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds


def seconds_to_timestring(total_seconds):
    # Create a timedelta object from the total seconds
    td = timedelta(seconds=total_seconds)
    # Extract days, hours, minutes, and seconds from timedelta
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    round(seconds)
    # Format the result as a string
    timestring = f"{days}T {hours}:{minutes}:{seconds}"
    return timestring


class GetStats:
    def __init__(self):
        # Initialize attributes for storing job statistics and calculations
        self.lost_gpu_time = None
        self.lost_cpu_time = None
        self.hostlist = None
        self.join_nodes = None
        self.real_time = None
        self.job_hostlist = None
        self.job_nodes_string = None
        self.gpu_eff = None
        self.job_gpu_nodes = None
        self.latest_end = ''
        self.jobs = None
        self.db_filter = None
        self.list_filter = None
        self.cores_job = 0
        self.job_id = 0
        self.cores = 0
        self.used_time = ''
        self.job_eff = 0
        self.job_steps = {}
        self.job_elapsed_cpu_time = None
        self.start = None
        self.end = None
        self.job_data = {}
        self.job_cpu = {}
        self.job_all = {}
        self.job_elapsed_s = 0
        self.total_cpu_time_sum = 0
        self.dict_steps = {}
        self.min_start = ''
        self.max_end = ''
        self.latest_avg_eff = ''
        self.avg_eff = 0
        self.intervall = ''
        self.nodelist = []
        self.all_nodes = []

    def job_stats(self, job_id: int) -> None:
        """
        Loads job data.py and calculates job statistics.
        """
        self.job_id = job_id
        self.job_data = pyslurm.db.Job.load(job_id)
        self.job_cpu = self.job_data.steps.to_dict()
        self.job_all = self.job_data.to_dict()
        self.job_elapsed_s = self.job_data.elapsed_time
        self.cores = self.job_data.cpus
        self.job_steps = count_keys_under_steps(self.job_all)
        self.all_nodes = gpu_node_data.hostlist_gpu()
        self.nodelist = self.job_data.nodelist
        self.hostlist = hostlist.expand_hostlist(self.nodelist)
        self.job_hostlist = [host + '.desy.de' for host in self.hostlist]
        set_nodes = set(self.all_nodes)
        self.job_gpu_nodes = ([node for node in self.job_hostlist if node in set_nodes]) if self.job_hostlist else None

        if self.job_gpu_nodes is not None:
            self.join_nodes = '|'.join([node for node in self.job_gpu_nodes])
            self.job_nodes_string = self.job_gpu_nodes if self.job_gpu_nodes is str else ' | '.join(self.job_gpu_nodes)

        # Calculate total CPU time used for job steps
        for step in self.job_steps:
            self.dict_steps[step] = self.job_cpu[step]["stats"]["total_cpu_time"]

        self.total_cpu_time_sum = round(sum(self.dict_steps.values()) / 1000)
        #  Calculate used time and booked time
        if self.job_elapsed_s:
            self.used_time = seconds_to_timestring(self.total_cpu_time_sum)
            self.real_time = seconds_to_timestring(self.job_elapsed_s)
            self.job_elapsed_cpu_time = seconds_to_timestring(self.job_elapsed_s * self.cores) if self.cores and self.job_elapsed_s else 0
            self.lost_cpu_time = seconds_to_timestring((self.job_elapsed_s * self.cores) - self.total_cpu_time_sum)

        # Format start and end times
        if self.job_data.end_time and self.job_data.start_time:
            self.start = datetime.utcfromtimestamp(self.job_data.start_time).strftime('%Y-%m-%dT%H:%M:%S')
            self.end = datetime.utcfromtimestamp(self.job_data.end_time).strftime('%Y-%m-%dT%H:%M:%S')

        # Calculate job efficiency

        self.calculate_efficiency()

    def calculate_efficiency(self) -> None:
        """
        Calculates the job efficiency as a percentage based on CPU time and elapsed time.
        """
        if self.cores is not None and self.job_elapsed_s is not None and self.cores > 0 and self.job_elapsed_s > 0:
            self.job_eff = round((self.total_cpu_time_sum / (self.cores * self.job_elapsed_s)) * 100, 1)
        else:
            self.job_eff = 0

    def get_jobs_calculate_insert_data(self, cur) -> None:
        """
        Fetches jobs, calculates their statistics, and inserts them into the database.
        """
        # Retrieve the highest jobID currently in the reportdata table
        # cur.execute("SELECT MAX(end) FROM reportdata")
        # self.latest_end = str(cur.fetchone()[0] or 0)
        self.latest_end = '2024-07-03T17:27:15'
        # Create a list of job IDs to filter and load jobs
        self.list_filter = round(time.time())
        # [self.jobID_count + i + 1 for i in range(1000)]
        self.db_filter = pyslurm.db.JobFilter(end_time=self.list_filter)
        self.jobs = pyslurm.db.Jobs.load(self.db_filter)
        # Process each job
        for job_id in self.jobs.keys():
            #try:
            stats = GetStats()
            stats.job_stats(job_id)
            data_dict = stats.to_dict()

            if data_dict['gpu_nodes'] is not None and data_dict['end'] is not None and data_dict['start'] is not None:
                #print(f"GPU-Data nodes = {data_dict['gpu_nodes']} end = {data_dict['end']} start = {data_dict['start']}")
                #print("get gpu data.py")
                stats.get_gpu_data()
                #print(self.job_hostlist)

            if stats.job_data.end_time is not None:
                end_time = datetime.fromtimestamp(stats.job_data.end_time)
                end_time = end_time.isoformat('T', 'auto')
                try:
                    if end_time is not None and end_time > self.latest_end:
                     #   print(f'execute query cause: {end_time} > {self.latest_end}  jobID: {job_id}')
                        data = stats.to_dict()
                        # Insert job statistics into reportdata table, avoiding conflicts on unique jobID
                        cur.execute("""
                                INSERT INTO reportdata (
                                    jobID, username, account, cpu_efficiency, lost_cpu_time, gpu_efficiency, lost_gpu_time, real_time, job_cpu_time,
                                    job_cpu_time_s, state, cores, gpu_nodes, start, end
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?) ON CONFLICT(jobID) DO UPDATE SET 
                                gpu_nodes = excluded.gpu_nodes,
                                lost_gpu_time = excluded.lost_gpu_time,
                                gpu_efficiency = excluded.gpu_efficiency 
                            """, (
                            data['job_id'], data['user'], data['account'], data['efficiency'], data['lost_cpu_time'], data['gpu_efficiency'],
                            data['lost_gpu_time'], data['real_time'], data['job_cpu_time'], data['job_cpu_time_s'], data['state'], data['cores'], data['gpu_nodes'],  data['start'], data['end']
                        ))
                        print(f"lost gpu time: {data['lost_gpu_time']}")
                        #    print(f"nodes: {data.py['gpu_efficiency']}")
                        cur.connection.commit()
                except Exception as e:
                    print(f"Error processing job {job_id}: {e}")
        #except Exception as err:
        #    print(f'Error endtime, job {job_id}:{err}')
        # Print an error message if job processing fails


    def calculate_avg_eff(self, cur) -> None:
        """
        Calculates and updates the average efficiency over time intervals.
        """
        # Retrieve the latest efficiency start time from the avg_eff table
        cur.execute("SELECT MAX(start) AS max_start FROM avg_eff")
        self.latest_avg_eff = cur.fetchone()[0] or self.min_start

        # Retrieve the minimum start time
        cur.execute("""
            SELECT MIN(start) AS min_start
            FROM reportdata 
            WHERE start IS NOT NULL AND start <> ''
        """)
        min_start = cur.fetchone()
        # Set the interval for calculating average efficiency
        self.intervall = min_start[0]

        # Loop through each time interval and calculate average efficiency
        while datetime.strptime(self.intervall, '%Y-%m-%dT%H:%M:%S') < datetime.now():
            interval_start = datetime.strptime(self.intervall, '%Y-%m-%dT%H:%M:%S')
            interval_end = interval_start + timedelta(hours=1)

            # Calculate average efficiency and count of jobs in the interval
            cur.execute("""
                SELECT AVG(efficiency) as a_eff, COUNT(cores) as c_job
                FROM reportdata 
                WHERE start <= ? AND end >= ?
            """, (interval_end, interval_start))
            a_eff, c_job = cur.fetchone()
            self.avg_eff = a_eff
            self.cores_job = c_job

            # Insert average efficiency into avg_eff table, avoiding conflicts on unique start times
            cur.execute(""" INSERT INTO avg_eff (eff, cores, start, end) VALUES (?, ?, ?, ?)
            ON CONFLICT(start) DO UPDATE SET eff = excluded.eff, cores = excluded.cores""",
                        (self.avg_eff, self.cores_job, self.intervall, interval_end.strftime('%Y-%m-%dT%H:%M:%S')))

            self.intervall = interval_end.strftime('%Y-%m-%dT%H:%M:%S')
            # print(self.intervall)
            cur.connection.commit()

        # Sleep for 2 seconds to avoid excessive querying

        return

    def get_gpu_data(self):
        step = 1
        prometheus_url = 'http://max-infra008.desy.de:9090/api/v1/query_range'
        params = {
            'query': f'nvidia_smi_utilization_gpu_ratio{{instance="{self.join_nodes}"}}',
            'start': f'{self.start}Z',
            'end': f'{self.end}Z',
            'step': f'{str(step)}m'
        }
        try:
            response = requests.get(prometheus_url, params=params)
            if response.status_code == 400:
                while True:
                    print("try")
                    step += 1
                    response = requests.get(prometheus_url, params=params)
                    print(response.raise_for_status())
                    if step == 5:
                        break
            response.raise_for_status()  # Raise an HTTPError if the response was unsuccessful
            data = response.json()
            # Debug: Print the full JSON response
            #print(f"Full JSON response: {data.py}")

            if 'data' in data and 'result' in data['data'] and len(data['data']['result']) > 0 and 'values' in data['data']['result'][0]:
                values = data['data']['result'][0]['values']
                int_values = [float(value[1]) for value in values]
                self.gpu_eff = (sum(int_values) / len(int_values)) if int_values else 0
                if self.job_gpu_nodes is not None and self.job_elapsed_s is not None:
                    lost_gpu_time_seconds = len(self.job_gpu_nodes) * self.job_elapsed_s * (1 - self.gpu_eff)
                    round(lost_gpu_time_seconds, 0)
                    print(f'round CPU : {lost_gpu_time_seconds}')
                    self.lost_gpu_time = str(timedelta(seconds=lost_gpu_time_seconds))
                    #print(f"gpu-usage: {self.gpu_eff}"))
            else:
                print(f"Error: Unexpected response structure{data}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
#        response = requests.get(prometheus_url, params=params)
#        print(f"gpu-usage: {response.json()['gpu_usage']}")
#        self.gpu_eff = response.json()['gpu_usage']

    def to_dict(self) -> dict:
        """
        Converts job statistics to a dictionary format
        """
        return {
            "job_id": self.job_id,
            "user": self.job_data.user_name,
            "account": self.job_data.account,
            "efficiency": self.job_eff,
            "lost_cpu_time": self.lost_cpu_time,
            "gpu_efficiency": self.gpu_eff * 100 if self.gpu_eff else None,
            "lost_gpu_time": self.lost_gpu_time,
            "real_time": self.real_time,
            "job_cpu_time": self.used_time,
            "job_cpu_time_s": self.job_elapsed_s,
            "state": self.job_data.state,
            "cores": self.cores,
            "gpu_nodes": self.job_nodes_string if self.job_nodes_string else None,
            "start": self.start,
            "end": self.end,
        }


class CreateFigures:
    def __init__(self, con):
        # Initialize the CreateFigures class with a database connection
        self.con = con

    def frame_user_all(self) -> None:
        """
        Displays all job data.py from the reportdata table in the Streamlit app.
        """
        df = pd.read_sql_query("""
            SELECT jobID, username, account, cpu_efficiency, lost_cpu_time, gpu_efficiency, lost_gpu_time, real_time, 
                   job_cpu_time, job_cpu_time_s AS realtime_in_s, state, cores, gpu_nodes, start, end 
            FROM reportdata
            """, self.con)
        st.write(df)

    def frame_group_by_user(self) -> None:
        """
        Displays average efficiency and job count grouped by username in the Streamlit app
        """
        lost_cpu = 0
        lost_gpu = 0
        start_date, end_date = st.date_input(
            'Start Date - End Date',
            [datetime.today() - timedelta(days=30),  datetime.today()],
        )
        if start_date and end_date:
            if start_date > end_date:
                st.error("Error: End date must fall after start date.")
            else:
                # Convert dates to string format for SQL query
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
        df = pd.read_sql_query(f"""
            SELECT username, AVG(cpu_efficiency) AS avg_cpu_efficiency, AVG(gpu_efficiency) AS avg_gpu_efficiency, 
                   lost_cpu_time, lost_gpu_time, COUNT(jobID) AS job_count
            FROM reportdata
            WHERE start >= '{start_date_str}' AND end <= '{end_date_str}'
            GROUP BY username
            """, con)
        for i in (df['lost_cpu_time']):
            if i is not None:
                lost_cpu += timestring_to_seconds(i)
        for j in (df['lost_gpu_time']):
            if j is not None:
                lost_gpu += timestring_to_seconds(j)

        lost_cpu = seconds_to_timestring(lost_cpu)
        lost_gpu = seconds_to_timestring(lost_gpu)
        df['lost_cpu_time'] = lost_cpu
        df['lost_gpu_time'] = lost_gpu
        st.write(df)

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
        df = pd.read_sql_query("""
            SELECT jobID, username, gpu_efficiency, 
                   cpu_efficiency, lost_cpu_time, lost_gpu_time, job_cpu_time_s, real_time, cores, state
            FROM reportdata
            ORDER BY job_cpu_time_s ASC;""", self.con)

        df['job_cpu_time_s'] = pd.to_numeric(df['job_cpu_time_s'], errors='coerce')
        df = df.dropna(subset=['job_cpu_time_s'])
        df['job_cpu_time_s'] = df['job_cpu_time_s'].astype(int)
        df['job_cpu_time_s'] = df['job_cpu_time_s'].apply(seconds_to_timestring)

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
#agsunset
    # Beispiel wie die Funktion aufgerufen werden könnte
    # scatter_chart_data_cpu_gpu_eff()



    # def scatter_chart_data(self):
    #     df = pd.read_sql_query("""
    #         SELECT jobID, username, gpu_efficiency, cpu_efficiency, lost_cpu_time, lost_gpu_time, job_cpu_time_s, real_time, cores, state
    #         FROM reportdata
    #         ORDER BY job_cpu_time_s ASC;""", self.con)
    #
    #     df['job_cpu_time_s'] = df['job_cpu_time_s'].apply(seconds_to_timestring)
    #
    #     # Create a new column to determine the color based on the presence of GPU efficiency
    #     df['color_scale'] = df['gpu_efficiency'].apply(lambda x: 'cpu' if pd.isna(x) else 'gpu')

    #     # Separate the data.py into two based on the new column
    #     df_cpu = df[df['color_scale'] == 'cpu']
    #     df_gpu = df[df['color_scale'] == 'gpu']
    #
    #     # Create scatter plots for both datasets
    #     fig = px.scatter(df_gpu, x="job_cpu_time_s", y="cpu_efficiency", color="gpu_efficiency",
    #                      color_continuous_scale="blues", size_max=1,
    #                      hover_data=["jobID", "username", "lost_cpu_time", "lost_gpu_time", "real_time", "cores", "state"])
    #
    #     fig_gpu = px.scatter(df_cpu, x="job_cpu_time_s", y="cpu_efficiency", color="cpu_efficiency",
    #                          color_continuous_scale="reds", size_max=1,
    #                          hover_data=["jobID", "username", "lost_cpu_time", "lost_gpu_time", "real_time", "cores", "state"])
    #
    #     # Update fig with fig_gpu traces
    #     for trace in fig_gpu['data.py']:
    #         fig.add_trace(trace)
    #
    #     st.plotly_chart(fig, theme=None)
if __name__ == "__main__":
    # Connect to SQLite database and create necessary tables
    con = sqlite3.connect('reports.db')
    cur = con.cursor()
    # # create table
    cur.execute("""
              CREATE TABLE IF NOT EXISTS reportdata (
                  jobID INTEGER NOT NULL UNIQUE,
                  username TEXT,
                  account TEXT,
                  cpu_efficiency REAL,
                  lost_cpu_time TEXT,
                  gpu_efficiency REAL,
                  lost_gpu_time TEXT,
                  real_time TEXT,
                  job_cpu_time TEXT,
                  job_cpu_time_s REAL,
                  state TEXT,
                  cores INT,
                  gpu_nodes TEXT,
                  start TEXT,
                  end TEXT
              )
              """)

    cur.execute("""CREATE TABLE IF NOT EXISTS avg_eff (eff REAL, count_job INT, start TEXT UNIQUE, end TEXT)""")

    cur.connection.commit()

    # Create figures and display them
    create = CreateFigures(con)
    create.frame_user_all()
    create.frame_group_by_user()
    #    create.chart_cpu_utilization()
    create.scatter_chart_data_cpu_gpu_eff()
    create.scatter_chart_data_color_lost_cpu()

    # Main loop to continuously fetch job data.py and update average efficiency
    while True:
        x = 29
        get = GetStats()
        get.get_jobs_calculate_insert_data(cur)
#        if x == 30:
#            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
#            get.calculate_avg_eff(cur)
#            x = 0
#        x += 1
