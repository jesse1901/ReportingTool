import pyslurm
import streamlit as st
import pandas as pd


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
        self.job_data = {}
        self.job_cpu = {}
        self.job_all = {}
        self.job_elapsed_s = 0
        self.cores = 0
        self.job_steps = {}
        self.job_state = ' '
        self.job_eff = 0
        self.dict_steps = {}
        self.total_cpu_sum = 0
        self.job_elapsed = 0
        self.job_list = []

    # Lädt die Jobdaten und berechnet die Jobstatistiken.
    def job_stats(self, job_id: int) -> list:
        self.job_id = job_id
        self.job_data = pyslurm.db.Job.load(job_id)
        self.job_cpu = self.job_data.steps.to_dict()
        self.job_all = self.job_data.to_dict()
        self.job_elapsed_s = self.job_data.elapsed_time
        self.cores = self.job_data.cpus
        self.job_steps = count_keys_under_steps(self.job_all)
        self.job_state = self.job_data.state

        # Berechnung vergangene Zeit
        self.job_elapsed = self.job_elapsed_s / 3600 if self.job_elapsed_s else 0

        # Auslesen gesamter CPU-Zeit für Jobsteps
        for i in self.job_steps:
            self.total_cpu_val = self.job_cpu[i]["stats"]["total_cpu_time"]
            self.dict_steps[i] = self.total_cpu_val

        self.total_cpu_sum = round(sum(self.dict_steps.values()) / 3600000, 3)

        self.calculate_efficiency()
        return self.job_list[self.job_id, self.cores, self.job_eff, self.job_state, self.job_steps, self.total_cpu_sum, self.job_eff]
    #Berechnent Effizienz
    def calculate_efficiency(self) -> None:

        if self.cores > 0 and self.job_elapsed > 0:
            self.job_eff = round((self.total_cpu_sum / (self.cores * self.job_elapsed)) * 100, 3)
        else:
            self.job_eff = 0

    def __repr__(self) -> list:
        return "(Job-ID='{0}', Cores='{1}', Effizienz='{2}', State='{3}, Steps='{4}', Cpu-Sum='{5}, Elapsed='{6})".format(
            self.job_id, self.cores, self.job_eff, self.job_state, self.job_steps, self.total_cpu_sum, self.job_eff, )


if __name__ == "__main__":

    jobs = [8201745, 8201746, 8201845, 8201846, 8201847, 8201848, 8201849, 8201850, 8201851, 8201852, 8201853, 8201854,
            8201855, 8201856, 8201857, 8201858, 8201859, 8201863, 8201864, 8201865, 8201866, 8201867, 8201868]
    job_eff_list = []

    for i in jobs:
        stats = GetStats()
        stats.job_stats(i)
        job_eff_list.extend([stats])

    st.write(pd.Dataframe({job_eff_list}))
