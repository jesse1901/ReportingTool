import pyslurm
import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import sqlite3
import plotly.express as px
import requests
import hostlist
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
        self.all_nodes = ['exflong03.desy.de', 'exflong04.desy.de', 'exflong05.desy.de', 'exflong06.desy.de',
                          'max-cfelg007.desy.de', 'max-cfelg007.desy.de', 'max-cmsg001.desy.de', 'max-cmsg002.desy.de',
                          'max-cmsg005.desy.de', 'max-cmsg006.desy.de', 'max-cmsg007.desy.de', 'max-cmsg008.desy.de',
                          'max-cmsg009.desy.de', 'max-cmsg010.desy.de', 'max-cssbg002.desy.de', 'max-cssbg002.desy.de',
                          'max-cssbg003.desy.de', 'max-cssbg003.desy.de', 'max-cssbg004.desy.de',
                          'max-cssbg004.desy.de', 'max-cssbg005.desy.de', 'max-cssbg005.desy.de',
                          'max-cssbg012.desy.de', 'max-cssbg012.desy.de', 'max-cssbg012.desy.de',
                          'max-cssbg012.desy.de', 'max-cssbg018.desy.de', 'max-cssbg018.desy.de',
                          'max-cssbg018.desy.de', 'max-cssbg018.desy.de', 'max-display001.desy.de',
                          'max-display001.desy.de', 'max-display001.desy.de', 'max-display001.desy.de',
                          'max-display002.desy.de', 'max-display002.desy.de', 'max-display002.desy.de',
                          'max-display002.desy.de', 'max-display003.desy.de', 'max-display003.desy.de',
                          'max-display003.desy.de', 'max-display003.desy.de', 'max-display004.desy.de',
                          'max-display004.desy.de', 'max-display004.desy.de', 'max-display004.desy.de',
                          'max-display005.desy.de', 'max-display005.desy.de', 'max-display005.desy.de',
                          'max-display005.desy.de', 'max-display007.desy.de', 'max-display007.desy.de',
                          'max-display008.desy.de', 'max-display008.desy.de', 'max-display009.desy.de',
                          'max-display009.desy.de', 'max-display010.desy.de', 'max-display010.desy.de',
                          'max-exfl-display001.desy.de', 'max-exfl-display001.desy.de', 'max-exfl-display002.desy.de',
                          'max-exfl-display002.desy.de', 'max-exfl-display003.desy.de', 'max-exfl-display003.desy.de',
                          'max-exfl-display004.desy.de', 'max-exfl-display004.desy.de', 'max-exflg005.desy.de',
                          'max-exflg005.desy.de', 'max-exflg006.desy.de', 'max-exflg007.desy.de',
                          'max-exflg009.desy.de', 'max-exflg010.desy.de', 'max-exflg011.desy.de',
                          'max-exflg012.desy.de', 'max-exflg013.desy.de', 'max-exflg014.desy.de',
                          'max-exflg015.desy.de', 'max-exflg016.desy.de', 'max-exflg017.desy.de',
                          'max-exflg018.desy.de', 'max-exflg019.desy.de', 'max-exflg020.desy.de',
                          'max-exflg021.desy.de', 'max-exflg022.desy.de', 'max-exflg023.desy.de',
                          'max-exflg024.desy.de', 'max-exflg025.desy.de', 'max-exflg026.desy.de',
                          'max-exflg027.desy.de', 'max-exflg028.desy.de', 'max-exflg028.desy.de',
                          'max-exflg029.desy.de',
                          'max-exflg029.desy.de', 'max-exflg030.desy.de', 'max-exflg030.desy.de',
                          'max-exflg031.desy.de', 'max-exflg031.desy.de', 'max-exflg032.desy.de',
                          'max-exflg033.desy.de', 'max-exflg034.desy.de', 'max-exflg035.desy.de',
                          'max-fs-display001.desy.de', 'max-fs-display001.desy.de', 'max-fs-display002.desy.de',
                          'max-fs-display002.desy.de', 'max-fs-display003.desy.de', 'max-fs-display003.desy.de',
                          'max-fs-display004.desy.de', 'max-fs-display004.desy.de', 'max-fs-display005.desy.de',
                          'max-fs-display005.desy.de', 'max-fs-display006.desy.de', 'max-fs-display006.desy.de',
                          'max-hzgg001.desy.de', 'max-hzgg001.desy.de', 'max-hzgg003.desy.de', 'max-hzgg003.desy.de',
                          'max-hzgg004.desy.de', 'max-hzgg004.desy.de', 'max-hzgg007.desy.de', 'max-hzgg007.desy.de',
                          'max-hzgg008.desy.de', 'max-hzgg008.desy.de', 'max-hzgg009.desy.de', 'max-hzgg009.desy.de',
                          'max-hzgg010.desy.de', 'max-hzgg010.desy.de', 'max-nova001.desy.de', 'max-nova001.desy.de',
                          'max-nova002.desy.de', 'max-nova002.desy.de', 'max-p3ag004.desy.de', 'max-p3ag005.desy.de',
                          'max-p3ag007.desy.de', 'max-p3ag008.desy.de', 'max-p3ag010.desy.de', 'max-p3ag011.desy.de',
                          'max-p3ag011.desy.de', 'max-p3ag011.desy.de', 'max-p3ag011.desy.de', 'max-p3ag012.desy.de',
                          'max-p3ag014.desy.de', 'max-p3ag015.desy.de', 'max-p3ag016.desy.de', 'max-p3ag017.desy.de',
                          'max-p3ag018.desy.de', 'max-p3ag019.desy.de', 'max-p3ag020.desy.de', 'max-p3ag021.desy.de',
                          'max-p3ag022.desy.de', 'max-p3ag023.desy.de', 'max-p3ag024.desy.de', 'max-p3ag025.desy.de',
                          'max-p3ag026.desy.de', 'max-p3ag027.desy.de', 'max-p3ag028.desy.de', 'max-p3ag029.desy.de',
                          'max-p3ag030.desy.de', 'max-p3ag031.desy.de', 'max-uhhg001.desy.de', 'max-uhhg001.desy.de',
                          'max-uhhg001.desy.de', 'max-uhhg001.desy.de', 'max-uhhg002.desy.de', 'max-uhhg002.desy.de',
                          'max-uhhg002.desy.de', 'max-uhhg002.desy.de', 'max-uhhg003.desy.de', 'max-uhhg003.desy.de',
                          'max-uhhg003.desy.de', 'max-uhhg003.desy.de', 'max-uhhg004.desy.de', 'max-uhhg004.desy.de',
                          'max-uhhg004.desy.de', 'max-uhhg004.desy.de', 'max-uhhg005.desy.de', 'max-uhhg005.desy.de',
                          'max-uhhg005.desy.de', 'max-uhhg005.desy.de', 'max-uhhg006.desy.de', 'max-uhhg006.desy.de',
                          'max-uhhg006.desy.de', 'max-uhhg006.desy.de', 'max-uhhg007.desy.de', 'max-uhhg007.desy.de',
                          'max-uhhg007.desy.de', 'max-uhhg007.desy.de', 'max-uhhg008.desy.de', 'max-uhhg008.desy.de',
                          'max-uhhg008.desy.de', 'max-uhhg008.desy.de', 'max-uhhg009.desy.de', 'max-uhhg009.desy.de',
                          'max-uhhg009.desy.de', 'max-uhhg009.desy.de', 'max-uhhg010.desy.de', 'max-uhhg010.desy.de',
                          'max-uhhg010.desy.de', 'max-uhhg010.desy.de', 'max-uhhg011.desy.de', 'max-uhhg011.desy.de',
                          'max-uhhg011.desy.de', 'max-uhhg011.desy.de', 'max-uhhg012.desy.de', 'max-uhhg012.desy.de',
                          'max-uhhg012.desy.de', 'max-uhhg012.desy.de', 'max-uhhg013.desy.de', 'max-uhhg013.desy.de',
                          'max-uhhg013.desy.de', 'max-uhhg013.desy.de', 'max-wng002.desy.de', 'max-wng002.desy.de',
                          'max-wng004.desy.de', 'max-wng005.desy.de', 'max-wng006.desy.de', 'max-wng007.desy.de',
                          'max-wng008.desy.de', 'max-wng008.desy.de', 'max-wng009.desy.de', 'max-wng009.desy.de',
                          'max-wng010.desy.de', 'max-wng012.desy.de', 'max-wng013.desy.de', 'max-wng014.desy.de',
                          'max-wng015.desy.de', 'max-wng016.desy.de', 'max-wng017.desy.de', 'max-wng018.desy.de',
                          'max-wng019.desy.de', 'max-wng020.desy.de', 'max-wng020.desy.de', 'max-wng021.desy.de',
                          'max-wng021.desy.de', 'max-wng022.desy.de', 'max-wng023.desy.de', 'max-wng037.desy.de',
                          'max-wng037.desy.de', 'max-wng037.desy.de', 'max-wng037.desy.de', 'max-wng038.desy.de',
                          'max-wng038.desy.de', 'max-wng038.desy.de', 'max-wng038.desy.de', 'max-wng039.desy.de',
                          'max-wng039.desy.de', 'max-wng039.desy.de', 'max-wng039.desy.de', 'max-wng040.desy.de',
                          'max-wng040.desy.de', 'max-wng040.desy.de', 'max-wng040.desy.de', 'max-wng041.desy.de',
                          'max-wng041.desy.de', 'max-wng041.desy.de', 'max-wng041.desy.de', 'max-wng042.desy.de',
                          'max-wng042.desy.de', 'max-wng042.desy.de', 'max-wng042.desy.de', 'max-wng061.desy.de',
                          'max-wng061.desy.de', 'max-wng061.desy.de', 'max-wng061.desy.de', 'max-wng062.desy.de',
                          'max-wng062.desy.de', 'max-wng062.desy.de', 'max-wng062.desy.de', 'max-wng063.desy.de',
                          'max-wng064.desy.de']

    def job_stats(self, job_id: int) -> None:
        """
        Loads job data and calculates job statistics.
        """
        self.job_id = job_id
        self.job_data = pyslurm.db.Job.load(job_id)
        self.job_cpu = self.job_data.steps.to_dict()
        self.job_all = self.job_data.to_dict()
        self.job_elapsed_s = self.job_data.elapsed_time
        self.cores = self.job_data.cpus
        self.job_steps = count_keys_under_steps(self.job_all)

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
            if self.job_elapsed_s:
                used_time_td = timedelta(seconds=self.total_cpu_time_sum)
                print("Total CPU time as timedelta:", used_time_td)

                # Convert the timedelta object to a pandas Timedelta
                td = pd.to_timedelta(used_time_td)
                print("Pandas Timedelta:", td)

                # Create a DataFrame with the timedelta object
                df = pd.DataFrame({'td': [td]})

                # Debugging: Check the DataFrame
                print("Initial DataFrame:\n", df)

                # Remove the days component from the timedelta
                df['td'] = df['td'] - pd.to_timedelta(df['td'].dt.days, unit='d')

                # Debugging: Check the DataFrame after removing days
                print("DataFrame after removing days:\n", df)

                # Assign the adjusted used time to self.used_time
                self.used_time = df['td'].iloc[0]
                print("Adjusted used time:", self.used_time)

                # Calculate and convert real time to a string
                self.real_time = str(timedelta(seconds=self.job_elapsed_s))
                print("Real time:", self.real_time)

                # Calculate and convert job elapsed CPU time to a string
                self.job_elapsed_cpu_time = str(timedelta(seconds=self.job_elapsed_s * self.cores))
                print("Job elapsed CPU time:", self.job_elapsed_cpu_time)

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
        if self.cores > 0 and self.job_elapsed_s > 0:
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
                #print("get gpu data")
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
                                    state, cores, gpu_nodes, start, end
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(jobID) DO UPDATE SET 
                                gpu_nodes = excluded.gpu_nodes,
                                gpu_efficiency = excluded.gpu_efficiency 
                            """, (
                            data['job_id'], data['user'], data['account'], data['efficiency'], data['lost_cpu_time'], data['gpu_efficiency'],
                            data['lost_gpu_time'], data['real_time'], data['job_cpu_time'], data['state'], data['cores'], data['gpu_nodes'],  data['start'], data['end']
                        ))
                    #    print(f"nodes: {data['gpu_nodes']}")
                    #    print(f"nodes: {data['gpu_efficiency']}")
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
        # print('sleep')
        # time.sleep(2)
        return

    def get_gpu_data(self):
        prometheus_url = 'http://max-infra008.desy.de:9090/api/v1/query_range'
        params = {
            'query': f'nvidia_smi_utilization_gpu_ratio{{instance="{self.join_nodes}"}}',
            'start': f'{self.start}Z',
            'end': f'{self.end}Z',
            'step': '1m'
        }
        try:
            response = requests.get(prometheus_url, params=params)
            response.raise_for_status()  # Raise an HTTPError if the response was unsuccessful
            data = response.json()
            # Debug: Print the full JSON response
            #print(f"Full JSON response: {data}")

            if 'data' in data and 'result' in data['data'] and len(data['data']['result']) > 0 and 'values' in data['data']['result'][0]:
                values = data['data']['result'][0]['values']
                int_values = [float(value[1]) for value in values]
                self.gpu_eff = (sum(int_values) / len(int_values)) if int_values else 0
                lost_gpu_time_seconds = len(self.job_gpu_nodes) * self.job_elapsed_s * (1 - self.gpu_eff)
                self.lost_gpu_time = str(timedelta(seconds=lost_gpu_time_seconds))
                #print(f"gpu-usage: {self.gpu_eff}"))
            else:
                print("Error: Unexpected response structure")
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
        Displays all job data from the reportdata table in the Streamlit app.
        """
        df = pd.read_sql_query("SELECT * FROM reportdata", self.con)
        st.write(df)

    def frame_group_by_user(self) -> None:
        """
        Displays average efficiency and job count grouped by username in the Streamlit app.
        """
        df = pd.read_sql_query("""
            SELECT username, AVG(efficiency) AS avg_efficiency, COUNT(jobID) AS anzahl_jobs 
            FROM reportdata 
            GROUP BY username""", self.con)
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

    def scatter_chart_data(self):
        df = pd.read_sql_query("SELECT jobID, gpu_efficiency, cpu_efficiency, lost_cpu_time, lost_gpu_time, job_cpu_time FROM reportdata ORDER BY lost_cpu_time ASC",
                               self.con)
        fig = px.scatter(df, x="job_cpu_time", y="cpu_efficiency", color="gpu_efficiency", size_max=1)
        st.plotly_chart(fig, theme=None)

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
    #    create.frame_group_by_user()
    #    create.chart_cpu_utilization()
    create.scatter_chart_data()

    # Main loop to continuously fetch job data and update average efficiency
    while True:
        x = 29
        get = GetStats()
        get.get_jobs_calculate_insert_data(cur)
#        if x == 30:
#            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
#            get.calculate_avg_eff(cur)
#            x = 0
#        x += 1
