import pyslurm
import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import sqlite3


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
        self.latest_end = None
        self.jobs = None
        self.db_filter = None
        self.list_filter = None
        self.cores_job = None
        self.job_id = 0
        self.cores = 0
        self.used_time = ''
        self.job_eff = 0
        self.job_steps = {}
        self.job_elapsed = ''
        self.start = ''
        self.end = ''
        self.job_data = {}
        self.job_cpu = {}
        self.job_all = {}
        self.job_elapsed_s = 0
        self.total_cpu_sum = 0
        self.dict_steps = {}
        self.min_start = ''
        self.max_end = ''
        self.latest_avg_eff = ''
        self.avg_eff = 0
        self.intervall = ''

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

        # Calculate total CPU time used for job steps
        for step in self.job_steps:
            self.dict_steps[step] = self.job_cpu[step]["stats"]["total_cpu_time"]

        self.total_cpu_sum = round(sum(self.dict_steps.values()) / 1000, 1)

        # Calculate used time and booked time
        if self.job_elapsed_s:
            self.used_time = str(timedelta(seconds=self.total_cpu_sum))
            self.job_elapsed = str(timedelta(seconds=self.job_elapsed_s))

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
            self.job_eff = round((self.total_cpu_sum / (self.cores * self.job_elapsed_s)) * 100, 1)
        else:
            self.job_eff = 0

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
            print(self.intervall)
            cur.connection.commit()

        # Sleep for 2 seconds to avoid excessive querying
        print('sleep')
        time.sleep(2)
        return

    def to_dict(self) -> dict:
        """
        Converts job statistics to a dictionary format.
        """
        return {
            "job_id": self.job_id,
            "user": self.job_data.user_name,
            "account": self.job_data.account,
            "efficiency": self.job_eff,
            "used": self.used_time,
            "booked": self.job_elapsed,
            "state": self.job_data.state,
            "cores": self.cores,
            "start": self.start,
            "end": self.end,
        }

    def get_jobs_calculate_insert_data(self, cur) -> None:
        """
        Fetches jobs, calculates their statistics, and inserts them into the database.
        """
        # Retrieve the highest jobID currently in the reportdata table
        cur.execute("SELECT MAX(end) FROM reportdata")
        self.latest_end = cur.fetchone()[0] or 0
        print(self.latest_end)

        # Create a list of job IDs to filter and load jobs
        self.list_filter = round(time.time())
# [self.jobID_count + i + 1 for i in range(1000)]
        self.db_filter = pyslurm.db.JobFilter(end_time=self.list_filter)
        self.jobs = pyslurm.db.Jobs.load(self.db_filter)

        # Process each job
        for job_id in self.jobs.keys():
            try:
                stats = GetStats()
                stats.job_stats(job_id)
                try:
                    if stats.job_data.end_time > self.latest_end:
                        data = stats.to_dict()
                        # Insert job statistics into reportdata table, avoiding conflicts on unique jobID
                        cur.execute("""
                            INSERT INTO reportdata (
                                jobID, username, account, efficiency, used_time, booked_time,
                                state, cores, start, end
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(jobID) DO NOTHING
                        """, (
                            data['job_id'], data['user'], data['account'], data['efficiency'],
                            data['used'], data['booked'], data['state'], data['cores'],
                            data['start'], data['end']
                        ))
                        print(data)
                        cur.connection.commit()
                except Exception as err:
                    print(f'Error endtime, job {job_id}:{err}')
            except Exception as e:
                # Print an error message if job processing fails
                print(f"Error processing job {job_id}: {e}")


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


if __name__ == "__main__":
    # Connect to SQLite database and create necessary tables
    con = sqlite3.connect('reports.db')
    cur = con.cursor()
########## NACH ID FILTERN NICHT ALLE JOBS
    # Create figures and display them
    create = CreateFigures(con)
    create.frame_user_all()
    create.frame_group_by_user()
    create.chart_cpu_utilization()

    # Main loop to continuously fetch job data and update average efficiency
    while True:
        x = 30
        get = GetStats()
        if x == 30:
            get.calculate_avg_eff(cur)
            x = 0
        get.get_jobs_calculate_insert_data(cur)
        x += 1



# create table
    # cur.execute("""
    #     CREATE TABLE IF NOT EXISTS reportdata (
    #         jobID INTEGER NOT NULL UNIQUE,
    #         username TEXT,
    #         account TEXT,
    #         efficiency REAL,
    #         used_time TEXT,
    #         booked_time TEXT,
    #         state TEXT,
    #         cores INT,
    #         start TEXT,
    #         end TEXT
    #     )
    # """)
    # cur.execute("""
    #     CREATE TABLE IF NOT EXISTS avg_eff (eff REAL, count_job INT, start TEXT UNIQUE, end TEXT)
    # """)
    # cur.fetchall()