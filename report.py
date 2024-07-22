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
        #var for calculate_avg_eff
        self.min_start = ''
        self.max_end = ''
        self.latest_avg_eff = ''
        self.avg_eff = []
        self.intervall = ''


        #var for get_jobs_calculate_insert_data
        self.jobID_count = 0
        self.list_filter = []
        self.db_filter = []
        self.jobs = {}
        self.keys = 0
        self.stats = []
        self.data = {}


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

        self.total_cpu_sum = round(sum(self.dict_steps.values()) / 1000, 1)

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
            self.job_eff = round((self.total_cpu_sum / (self.cores * self.job_elapsed_s)) * 100, 1)
        else:
            self.job_eff = 0


    def calculate_avg_eff(self) -> None:
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS avg_eff ( eff INT, start TEXT, end TEXT)
                    """)
        self.latest_avg_eff = cur.execute("SELECT MAX(start) AS max_start FROM avg_eff")
        self.min_start = cur.execute("SELECT MIN(start) AS min_start FROM reportdata WHERE start <> '' AND start IS NOT NULL")
        self.max_end = self.min_start = cur.execute("SELECT MAX(end) AS min_start FROM reportdata WHERE start <> '' AND start IS NOT NULL")

        self.intervall = self.min_start if self.min_start > self.latest_avg_eff else self.latest_avg_eff

        while self.intervall < datetime.now():
            self.avg_eff = cur.execute("SELECT AVG(efficiency) FROM reportdata WHERE start <= ? AND end >= ?",
                                   (self.self.intervall, datetime.strptime(self.intervall, '%Y-%m-%dT%H:%M:%S') + timedelta(hours=1)))

            cur.execute("INSERT INTO reportdata( jobID, start, end )VALUES (?,?,?)",
                    (self.avg_eff, self.intervall, datetime.strptime(self.intervall, '%Y-%m-%dT%H:00:00') + timedelta(hours=1)))
            self.intervall = datetime.strptime(self.intervall, '%Y-%m-%dT%H:00:00') + timedelta(hours=1)
            print(self.intervall)

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

    def get_jobs_calculate_insert_data(self):
        cur.execute("""
                    SELECT MAX(jobID) FROM reportdata
        """)
        x = cur.fetchall()
        self.jobID_count = x[0][0]
        print(self.jobID_count)
        for i in range(500):
            self.jobID_count += 1
            self.list_filter.append(self.jobID_count)
        self.db_filter = pyslurm.db.JobFilter(ids=self.list_filter)
        self.jobs = pyslurm.db.Jobs.load(self.db_filter)

        for self.keys in self.jobs.keys():
            try:
                self.stats = GetStats()
                self.stats.job_stats(self.keys)
                if self.stats.job_data.end_time is not None:
                    self.data = self.stats.to_dict()
                    cur.execute("""
                        INSERT INTO reportdata (
                            jobID, username, account, efficiency, used_time, booked_time,
                            state, cores, start, end
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(jobID) DO NOTHING
                    """, (
                        self.data['job_id'], self.data['user'], self.data['account'], self.data['efficiency'],
                        self.data['used'], self.data['booked'], self.data['state'], self.data['cores'],
                        self.data['start'], self.data['end']))
                    con.commit()
            except:
                print("error")



class CreateFigures:
    def __init__(self):
        self.frame_all = ''
        self.frame_group_user = ''
        self.chart_cpu_u = ''

    def frame_user_all(self):
        self.frame_all = pd.read_sql_query("SELECT * FROM reportdata", con)
        st.write(pd.DataFrame(self.frame_all))

    def frame_group_by_user(self):
        self.frame_group_user = pd.read_sql_query(
            "SELECT username, AVG(efficiency) AS avg_efficiency, COUNT(jobID) AS anzahl_jobs FROM reportdata GROUP BY username",
            con)
        st.write(pd.DataFrame(self.frame_group_user))

    # def chart_cpu_utilization(self):
    #     self.chart_cpu_u = pd.read_sql_query("""
    #         SELECT strftime('%Y-%m-%d %H:00:00', start) AS period, AVG(efficiency) AS avg_efficiency
    #         FROM reportdata
    #         GROUP BY strftime('%Y-%m-%d %H:00:00', start)
    #         ORDER BY period""", con)
    #     st.line_chart(pd.DataFrame(self.chart_cpu_u).set_index('period'))



if __name__ == "__main__":
    #Database connection
    con = sqlite3.connect('reports.db') #sqlite connection
    cur = con.cursor()
    cur.execute("""
                CREATE TABLE IF NOT EXISTS reportdata (
                jobID INTEGER NOT NULL UNIQUE, username TEXT, account TEXT, efficiency REAL, used_time TEXT,
                booked_time TEXT, state TEXT, cores INT, start TEXT, end TEXT  )""")

    create = CreateFigures()

    create.frame_user_all()
    create.frame_user_all()
   # create.chart_cpu_utilization()

    while True:
        get.calculate_avg_eff()
    while True:
        get = GetStats()
        get.get_jobs_calculate_insert_data()



