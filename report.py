import pyslurm
import streamlit as st
import pandas as pd
import time
from datetime import timedelta, datetime
import sqlite3

def count_keys_under_steps(d):
    steps_dict = d.get('steps', {})
    if isinstance(steps_dict, dict):
        return list(steps_dict.keys())
    return []

class GetStats:
    def __init__(self):
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
        self.count_job = 0
    def job_stats(self, job_id: int) -> None:
        self.job_id = job_id
        self.job_data = pyslurm.db.Job.load(job_id)
        self.job_cpu = self.job_data.steps.to_dict()
        self.job_all = self.job_data.to_dict()
        self.job_elapsed_s = self.job_data.elapsed_time
        self.cores = self.job_data.cpus
        self.job_steps = count_keys_under_steps(self.job_all)

        for step in self.job_steps:
            self.dict_steps[step] = self.job_cpu[step]["stats"]["total_cpu_time"]

        self.total_cpu_sum = round(sum(self.dict_steps.values()) / 1000, 1)

        if self.job_elapsed_s:
            self.used_time = str(timedelta(seconds=self.total_cpu_sum))
            self.job_elapsed = str(timedelta(seconds=self.job_elapsed_s))

        if self.job_data.end_time and self.job_data.start_time:
            self.start = datetime.utcfromtimestamp(self.job_data.start_time).strftime('%Y-%m-%dT%H:%M:%S')
            self.end = datetime.utcfromtimestamp(self.job_data.end_time).strftime('%Y-%m-%dT%H:%M:%S')

        self.calculate_efficiency()

    def calculate_efficiency(self) -> None:
        if self.cores > 0 and self.job_elapsed_s > 0:
            self.job_eff = round((self.total_cpu_sum / (self.cores * self.job_elapsed_s)) * 100, 1)
        else:
            self.job_eff = 0

    def calculate_avg_eff(self, cur) -> None:
        # Get the latest average efficiency start
        cur.execute("SELECT MAX(start) AS max_start FROM avg_eff")
        self.latest_avg_eff = cur.fetchone()[0] or self.min_start

        cur.execute("""
            SELECT MIN(start) AS min_start, MAX(end) AS max_end 
            FROM reportdata 
            WHERE start IS NOT NULL AND start <> ''
        """)
        min_start, max_end = cur.fetchone()
        self.min_start = min_start
        self.max_end = max_end

        self.intervall = self.min_start if not self.latest_avg_eff or self.min_start > self.latest_avg_eff else self.latest_avg_eff
        print(self.intervall)

        while datetime.strptime(self.intervall, '%Y-%m-%dT%H:%M:%S') + timedelta(hours=1) < datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):
            interval_start = datetime.strptime(self.intervall, '%Y-%m-%dT%H:%M:%S')
            interval_end = interval_start + timedelta(hours=1)

            cur.execute("""
                SELECT AVG(efficiency)  
                FROM reportdata 
                WHERE start <= ? AND end >= ?
            """, (interval_end, interval_start))
            self.avg_eff = cur.fetchone()[0]

            cur.execute(""" SELECT COUNT(job_id) 
                FROM reportdata """)
            self.count_job = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO avg_eff (eff, job_count, start, end)
                VALUES (?, ?, ?, ?) ON CONFLICT(start) DO NOTHING
            """, (self.avg_eff, self.count_job, self.intervall, interval_end.strftime('%Y-%m-%dT%H:%M:%S')))
            cur.connection.commit()
            self.intervall = interval_end.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            time.sleep(2)
            return

    def to_dict(self) -> dict:
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
        cur.execute("SELECT MAX(jobID) FROM reportdata")
        self.jobID_count = cur.fetchone()[0] or 0
        print(self.jobID_count)
        self.list_filter = [self.jobID_count + i + 1 for i in range(100)]
        self.db_filter = pyslurm.db.JobFilter(ids=self.list_filter)
        self.jobs = pyslurm.db.Jobs.load(self.db_filter)

        for job_id in self.jobs.keys():
            try:
                stats = GetStats()
                stats.job_stats(job_id)
                if stats.job_data.end_time:
                    data = stats.to_dict()
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
                    cur.connection.commit()
                    print(f'inserted{data["job_id"]}')
            except Exception as e:
                print(f"Error processing job {job_id}: {e}")

class CreateFigures:
    def __init__(self, con):
        self.con = con

    def frame_user_all(self) -> None:
        df = pd.read_sql_query("SELECT * FROM reportdata", self.con)
        st.write(df)

    def frame_group_by_user(self) -> None:
        df = pd.read_sql_query("""
            SELECT username, AVG(efficiency) AS avg_efficiency, COUNT(jobID) AS anzahl_jobs 
            FROM reportdata 
            GROUP BY username
        """, self.con)
        st.write(df)

    def chart_cpu_utilization(self) -> None:
        df = pd.read_sql_query("""
            SELECT strftime('%Y-%m-%d %H:00:00', start) AS period, eff AS avg_efficiency
            FROM avg_eff
            GROUP BY strftime('%Y-%m-%d %H:00:00', start)
            ORDER BY period
        """, self.con)
        st.line_chart(df.set_index('period'))

if __name__ == "__main__":
    con = sqlite3.connect('reports.db')
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reportdata (
            jobID INTEGER NOT NULL UNIQUE, 
            username TEXT, 
            account TEXT, 
            efficiency REAL, 
            used_time TEXT,
            booked_time TEXT, 
            state TEXT, 
            cores INT, 
            start TEXT, 
            end TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS avg_eff (eff REAL, count_job INT, start TEXT UNIQUE, end TEXT)
    """)
    cur.fetchall()

    create = CreateFigures(con)
    create.frame_user_all()
    create.frame_group_by_user()
    create.chart_cpu_utilization()

    while True:
        get = GetStats()
        get.get_jobs_calculate_insert_data(cur)
        get.calculate_avg_eff(cur)
