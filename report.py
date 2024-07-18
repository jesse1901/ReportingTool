import pyslurm
import streamlit as st
import pandas as pd
import mysql.connector
from datetime import timedelta, datetime


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
        self.job_state = ''
        self.used_time = ''
        self.job_eff = 0
        self.job_steps = {}
        self.job_elapsed = 0
        self.end = 0

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
        self.job_state = self.job_data.state
        self.job_cpu = self.job_data.steps.to_dict()
        self.job_all = self.job_data.to_dict()
        self.job_elapsed_s = self.job_data.elapsed_time
        self.cores = self.job_data.cpus
        self.job_steps = count_keys_under_steps(self.job_all)
        self.end = self.job_data.end_time

        # Auslesen gesamter CPU-Zeit für Job steps
        for i in self.job_steps:
            self.dict_steps[i] = self.job_cpu[i]["stats"]["total_cpu_time"]

        self.total_cpu_sum = round(sum(self.dict_steps.values()) / 1000, 3)
        if self.job_elapsed_s is not None:
            self.used_time = str(timedelta(seconds=self.total_cpu_sum))
            self.job_elapsed = str(timedelta(seconds=self.job_elapsed_s))

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
            "cores": self.cores,
            "job_efficiency": self.job_eff,
            "job_state": self.job_state,
            "job_steps": self.job_steps,
            "genutzte Zeit": self.used_time,
            "gebuchte Zeit": self.job_elapsed,
            "end": self.end,
        }


if __name__ == "__main__":
    db_filter = pyslurm.db.JobFilter()
    jobs = pyslurm.db.Jobs.load(db_filter)

    job_eff_list = []
    for keys in jobs.keys():
        stats = GetStats()
        stats.job_stats(keys)
        if stats.end is not None:
            job_eff_list.append(stats.to_dict())
            print(job_eff_list)

#    df = pd.DataFrame(job_eff_list)
#    st.write(df)
