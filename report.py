import pyslurm
import streamlit as st
import pandas as pd
from datetime import timedelta, datetime
import sqlite3



# Zählt die Schlüssel unter dem 'steps'-Schlüssel in einem gegebenen Dictionary.
def count_keys_under_steps(d):
    steps_dict = d.get('steps', {})
    if isinstance(steps_dict, dict):
        return list(steps_dict.keys())
    return []


class GetStats:

    # Initialisiert die Klasse GetStats mit den notwendigen Attributen.
    def __init__(self):
        self.job_id = 0
        self.cores = 0
        self.used_time = 0
        self.job_eff = 0
        self.job_steps = {}
        self.job_elapsed = 0
        self.start = ''
        self.end = ''

        self.job_data = {}
        self.job_cpu = {}
        self.job_all = {}
        self.job_elapsed_s = 0
        self.total_cpu_sum = 0
        self.dict_steps = {}
        self.job_list = []

    # Lädt die Jobdaten und berechnet die Jobstatistiken.
    def job_stats(self, job_id: int) -> None:
        self.job_id = job_id
        self.job_data = pyslurm.db.Job.load(job_id)
        self.job_cpu = self.job_data.steps.to_dict()
        self.job_all = self.job_data.to_dict()
        self.job_elapsed_s = self.job_data.elapsed_time
        self.cores = self.job_data.cpus
        self.job_steps = count_keys_under_steps(self.job_all)

        # Auslesen gesamter CPU-Zeit für Job steps
        for i in self.job_steps:
            self.dict_steps[i] = self.job_cpu[i]["stats"]["total_cpu_time"]

        self.total_cpu_sum = round(sum(self.dict_steps.values()) / 1000, 3)

        if self.job_elapsed_s and self.used_time is not None:
            self.used_time = str(timedelta(seconds=self.total_cpu_sum))
            self.job_elapsed = str(timedelta(seconds=self.job_elapsed_s))

        if self.job_data.end_time and self.job_data.start_time is not None:
            self.start = datetime.utcfromtimestamp(self.job_data.start_time).strftime('%Y-%m-%dT%H:%M:%S')
            self.end = datetime.utcfromtimestamp(self.job_data.end_time).strftime('%Y-%m-%dT%H:%M:%S')

        self.calculate_efficiency()

    # Berechnet Effizienz
    def calculate_efficiency(self) -> None:

        if self.cores > 0 and self.job_elapsed_s > 0:
            self.job_eff = round((self.total_cpu_sum / (self.cores * self.job_elapsed_s)) * 100, 3)
        else:
            self.job_eff = 0

    #def check_state(self) -> bool:

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

if __name__ == "__main__":
    #Database connection
    con = sqlite3.connect('reports.db') #sqlite connection
    cur = con.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS reportdata (
                jobID INTEGER NOT NULL UNIQUE, username TEXT, account TEXT, efficiency REAL, used_time TEXT,
                booked_time TEXT, state TEXT, cores INT, start TEXT, end TEXT  )""")

    conn_streamlit = st.connection('reports_db', type='sql')

    x = conn_streamlit.query("SELECT * FROM reportdata", ttl=600)
    df = pd.DataFrame(x)
    st.write(df)

    while True:

        cur.execute("""
                    SELECT MAX(jobID) FROM reportdata
        """)
        x = cur.fetchall()
        jobID = x[0][0]
        print(jobID)
        list_filter = []
        for i in range(500):
            jobID += 1
            list_filter.append(jobID)
        db_filter = pyslurm.db.JobFilter(ids=list_filter)
        jobs = pyslurm.db.Jobs.load(db_filter)

        for keys in jobs.keys():
            try:
                stats = GetStats()
                stats.job_stats(keys)
                if stats.job_data.end_time is not None:
                    data = stats.to_dict()
                    cur.execute("""
                        INSERT INTO reportdata (
                            jobID, username, account, efficiency, used_time, booked_time,
                            state, cores, start, end
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(jobID) DO NOTHING
                    """, (
                        data['job_id'], data['user'], data['account'], data['efficiency'],
                        data['used'], data['booked'], data['state'], data['cores'],
                        data['start'], data['end']))
                    con.commit()
            except:
                print("error")