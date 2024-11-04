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



def timestring_to_seconds(timestring):
    if pd.isna(timestring) or timestring == '0' or timestring == 0:
        return 0

    if isinstance(timestring, float):
        timestring = str(int(timestring))  # Convert float to integer string

    # Ensure timestring is in string format
    timestring = str(timestring).strip()

    # Split by 'T' to separate days from time
    if 'T' in timestring:
        days_part, time_part = timestring.split('T')
    else:
        days_part, time_part = '0', timestring

    # Convert days part
    days = int(days_part.strip()) if days_part.strip() else 0

    # Convert time part (HH:MM:SS)
    time_parts = time_part.split(':')
    hours = int(time_parts[0].strip()) if len(time_parts) > 0 else 0
    minutes = int(time_parts[1].strip()) if len(time_parts) > 1 else 0
    seconds = int(time_parts[2].strip()) if len(time_parts) > 2 else 0

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
    def __init__(_self):
        # Initialize attributes for storing job statistics and calculations
        _self.partition = None
        _self.name = None
        _self.lost_gpu_time_sec = None
        _self.lost_gpu_time_seconds = None
        _self.lost_gpu_time = None
        _self.lost_cpu_time = None
        _self.hostlist = None
        _self.join_nodes = None
        _self.real_time = None
        _self.job_hostlist = None
        _self.job_nodes_string = None
        _self.gpu_eff = None
        _self.job_gpu_nodes = None
        _self.latest_end = ''
        _self.jobs = None
        _self.db_filter = None
        _self.list_filter = None
        _self.cores_job = 0
        _self.job_id = 0
        _self.cores = 0
        _self.used_time = ''
        _self.job_eff = 0
        _self.job_steps = {}
        _self.job_elapsed_cpu_time = None
        _self.start = None
        _self.end = None
        _self.job_data = {}
        _self.job_cpu = {}
        _self.job_all = {}
        _self.job_elapsed_s = 0
        _self.total_cpu_time_sum = 0
        _self.dict_steps = {}
        _self.min_start = ''
        _self.max_end = ''
        _self.latest_avg_eff = ''
        _self.avg_eff = 0
        _self.intervall = ''
        _self.nodelist = []
        _self.all_nodes = []

    def job_stats(_self, job_id: int) -> None:
        """
        Loads job data.py and calculates job statistics.
        """
        _self.job_id = job_id
        _self.job_data = pyslurm.db.Job.load(job_id)
        _self.job_cpu = _self.job_data.steps.to_dict()
        _self.job_all = _self.job_data.to_dict()
        _self.job_elapsed_s = _self.job_data.elapsed_time
        _self.cores = _self.job_data.cpus
        _self.name = _self.job_data.name
        _self.partition = _self.job_data.partition
        _self.job_steps = count_keys_under_steps(_self.job_all)
        _self.all_nodes = gpu_node_data.hostlist_gpu()
        _self.nodelist = _self.job_data.nodelist
        _self.hostlist = hostlist.expand_hostlist(_self.nodelist)
        _self.job_hostlist = [host + '.desy.de' for host in _self.hostlist]
        set_nodes = set(_self.all_nodes)
        _self.job_gpu_nodes = ([node for node in _self.job_hostlist if node in set_nodes]) if _self.job_hostlist else None

        if _self.job_gpu_nodes is not None:
            _self.join_nodes = '|'.join([node for node in _self.job_gpu_nodes])
            _self.job_nodes_string = _self.job_gpu_nodes if _self.job_gpu_nodes is str else ' | '.join(_self.job_gpu_nodes)

        # Calculate total CPU time used for job steps
        for step in _self.job_steps:
            _self.dict_steps[step] = _self.job_cpu[step]["stats"]["total_cpu_time"]

        _self.total_cpu_time_sum = round(sum(_self.dict_steps.values()) / 1000)
        #  Calculate used time and booked time
        if _self.job_elapsed_s:
            _self.used_time = seconds_to_timestring(_self.total_cpu_time_sum)
            _self.real_time = seconds_to_timestring(_self.job_elapsed_s)
            _self.job_elapsed_cpu_time = seconds_to_timestring(
                _self.job_elapsed_s * _self.cores) if _self.cores and _self.job_elapsed_s else 0
            _self.lost_cpu_time = seconds_to_timestring((_self.job_elapsed_s * _self.cores) - _self.total_cpu_time_sum)

        # Format start and end times
        if _self.job_data.end_time and _self.job_data.start_time:
            _self.start = datetime.utcfromtimestamp(_self.job_data.start_time).strftime('%Y-%m-%dT%H:%M:%S')
            _self.end = datetime.utcfromtimestamp(_self.job_data.end_time).strftime('%Y-%m-%dT%H:%M:%S')

        # Calculate job efficiency

        _self.calculate_efficiency()

    def calculate_efficiency(_self) -> None:
        """
        Calculates the job efficiency as a percentage based on CPU time and elapsed time.
        """
        if _self.cores is not None and _self.job_elapsed_s is not None and _self.cores > 0 and _self.job_elapsed_s > 0:
            _self.job_eff = round((_self.total_cpu_time_sum / (_self.cores * _self.job_elapsed_s)) * 100, 1)
        else:
            _self.job_eff = 0

    def get_jobs_calculate_insert_data(_self, cur) -> None:
        """
        Fetches jobs, calculates their statistics, and inserts them into the database.
        """
        # Retrieve the highest jobID currently in the reportdata table
        # cur.execute("SELECT MAX(end) FROM reportdata")
        # _self.latest_end = str(cur.fetchone()[0] or 0)
        jobs = cur.execute("""
                    SELECT jobID FROM reportdata WHERE partition IS NULL
        """)
        jobs = cur.fetchall()
        count = 1
        # Process each job
        for job_id in jobs:
            #try:
            stats = GetStats()
            stats.job_stats(job_id[0])
            data_dict = stats.to_dict()

            if data_dict['gpu_nodes'] is not None and data_dict['end'] is not None and data_dict['start'] is not None:
                #print(f"GPU-Data nodes = {data_dict['gpu_nodes']} end = {data_dict['end']} start = {data_dict['start']}")
                #print("get gpu data.py")
                stats.get_gpu_data()
                #print(_self.job_hostlist)

            if stats.job_data.end_time is not None:
                end_time = datetime.fromtimestamp(stats.job_data.end_time)
                end_time = end_time.isoformat('t', 'auto')
                try:
                    if end_time is not None and end_time > _self.latest_end:
                        #   print(f'execute query cause: {end_time} > {_self.latest_end}  jobID: {job_id}')
                        data = stats.to_dict()

                        # Insert job statistics into reportdata table, avoiding conflicts on unique jobID
                        lost_gpu_time_sec = int(data['lost_gpu_time_sec']) if data['lost_gpu_time_sec'] else None
                        lost_cpu_time_sec = int(data['lost_cpu_time_sec']) if data['lost_cpu_time_sec'] else None

                        # Insert job statistics into reportdata table, avoiding conflicts on unique jobID
                        cur.execute("""
                                          INSERT INTO reportdata (
                                              jobID, username, account, cpu_efficiency, lost_cpu_time, lost_cpu_time_sec, 
                                              gpu_efficiency, lost_gpu_time, lost_gpu_time_sec, real_time, job_cpu_time,
                                              real_time_sec, state, cores, gpu_nodes, start, end, job_name, 
                                              total_cpu_time_booked, partition
                                          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) 
                                          ON CONFLICT(jobID) DO UPDATE SET 
                                              gpu_nodes = excluded.gpu_nodes,
                                              lost_gpu_time = excluded.lost_gpu_time,
                                              gpu_efficiency = excluded.gpu_efficiency,
                                              lost_gpu_time_sec = excluded.lost_gpu_time_sec,
                                              lost_cpu_time_sec = excluded.lost_cpu_time_sec,
                                              job_name = excluded.job_name,
                                              total_cpu_time_booked = excluded.total_cpu_time_booked,
                                              partition = excluded.partition
                                      """, (
                            data['job_id'], data['user'], data['account'], data['efficiency'], data['lost_cpu_time'],
                            lost_cpu_time_sec, data['gpu_efficiency'], data['lost_gpu_time'], lost_gpu_time_sec,
                            data['real_time'], data['job_cpu_time'], data['real_time_sec'], data['state'],
                            data['cores'], data['gpu_nodes'], data['start'], data['end'], data['job_name'],
                            data['total_cpu_time_booked'], data['partition']
                        ))
                        print(f"inserted Job {count}: ID = {data['job_id']}")
                        count += 1
                        #print(f"lost gpu time sec: {data['lost_gpu_time_sec']}")
                        cur.connection.commit()
                except Exception as e:
                    print(f"Error processing job {job_id}: {e}")
        #except Exception as err:
        #    print(f'Error endtime, job {job_id}:{err}')
        # Print an error message if job processing fails

    def get_gpu_data(_self):
        step = 1
        prometheus_url = 'http://max-infra008.desy.de:9090/api/v1/query_range'
        params = {
            'query': f'nvidia_smi_utilization_gpu_ratio{{instance="{_self.join_nodes}"}}',
            'start': f'{_self.start}Z',
            'end': f'{_self.end}Z',
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

            if 'data' in data and 'result' in data['data'] and len(data['data']['result']) > 0 and 'values' in \
                    data['data']['result'][0]:
                values = data['data']['result'][0]['values']
                int_values = [float(value[1]) for value in values]
                _self.gpu_eff = (sum(int_values) / len(int_values)) if int_values else 0
                if _self.job_gpu_nodes is not None and _self.job_elapsed_s is not None:
                    _self.lost_gpu_time_seconds = int(
                        (len(_self.job_gpu_nodes) * _self.job_elapsed_s * (1 - _self.gpu_eff)))
                    _self.lost_gpu_time_sec = _self.lost_gpu_time_seconds
                    _self.lost_gpu_time = str(timedelta(seconds=_self.lost_gpu_time_sec))
            else:
                print(f"Error: Unexpected response structure{data}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    #        response = requests.get(prometheus_url, params=params)
    #        print(f"gpu-usage: {response.json()['gpu_usage']}")
    #        _self.gpu_eff = response.json()['gpu_usage']

    def to_dict(_self) -> dict:
        """
        Converts job statistics to a dictionary format
        """
        return {
            "job_id": _self.job_id,
            "user": _self.job_data.user_name,
            "account": _self.job_data.account,
            "efficiency": _self.job_eff,
            "lost_cpu_time": _self.lost_cpu_time,
            "lost_cpu_time_sec": timestring_to_seconds(_self.lost_cpu_time),
            "gpu_efficiency": _self.gpu_eff * 100 if _self.gpu_eff else None,
            "lost_gpu_time": _self.lost_gpu_time,
            "lost_gpu_time_sec": _self.lost_gpu_time_sec,
            "real_time": _self.real_time,
            "job_cpu_time": _self.used_time,
            "real_time_sec": _self.job_elapsed_s,
            "state": _self.job_data.state,
            "cores": _self.cores,
            "gpu_nodes": _self.job_nodes_string if _self.job_nodes_string else None,
            "start": _self.start,
            "end": _self.end,
            "job_name": _self.name,
            "total_cpu_time_booked": _self.job_elapsed_cpu_time,
            "partition": _self.partition
        }

if __name__ == "__main__":
    # Connect to SQLite database and create necessary tables
    con = sqlite3.connect('reports.db')
    cur = con.cursor()

    while True:
        get = GetStats()
        get.get_jobs_calculate_insert_data(cur)

