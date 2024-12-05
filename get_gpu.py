import requests
from datetime import time, timedelta, datetime, timezone
import sqlite3
import hostlist
import gpu_node_data


def get_rows_without_gpu(con):
    cur = con.cursor()
    step = 1
    while True:
        try:
            cur.execute("""SELECT Start, End, NodeList, Elapsed, JobID 
                FROM slurm 
                    WHERE 
                (NGPUS > 0 AND GpuUtil IS NULL AND Start IS NOT NULL AND End IS NOT NULL AND State != 'RUNNING') 
                     OR 
                (ReqGPUS IS NULL AND Start IS NOT NULL AND End IS NOT NULL AND State != 'RUNNING')""")
            
            data = cur.fetchall()
            if not data:
                break
            for row in data:
                if row[0] and row[1]:
                    get_gpu_data(cur, row, step)
            step += 1
            if step > 10:
                print(f"no response for {len(data)} jobs.")
                break
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

        
def get_gpu_data(cur, row, step):
    req_count = 1
    start = datetime.fromtimestamp(row[0]).strftime('%Y-%m-%dT%H:%M:%S')
    end = datetime.fromtimestamp(row[1]).strftime('%Y-%m-%dT%H:%M:%S')
    nodelist = row[2]
    Elapsed = row[3]
    jobID = row[4]
    nodelist = hostlist.expand_hostlist(nodelist)
    nodelist = [host + '.desy.de' for host in nodelist]
    
    all_gpu_nodes = gpu_node_data.hostlist_gpu()
    
    gpu_nodes = [node for node in nodelist if any(node == gpu_node[0] for gpu_node in all_gpu_nodes)]
    total_gpus = sum(gpu_count for node, gpu_count in all_gpu_nodes if node in gpu_nodes)
    
    gpu_eff = None
    
    if gpu_nodes:
        gpu_available = total_gpus
    else:
        gpu_available = 0
    
    if gpu_nodes:
        join_nodes = '|'.join([node for node in gpu_nodes])
        step = step
        prometheus_url = 'http://max-infra008.desy.de:9090/api/v1/query_range'
        params = {
            'query': f'nvidia_smi_utilization_gpu_ratio{{instance=~"{join_nodes}"}}',
            'start': f'{start}Z',
            'end': f'{end}Z',
            'step': f'{str(step)}m'
        }
        try:
            response = requests.get(prometheus_url, params=params)
            response.raise_for_status() 
            response = response.json()

            if 'data' in response and 'result' in response['data'] and len(response['data']['result']) > 0:
                total_sum = 0
                value_count = 0

                for result in response['data']['result']:
                    values = result['values']
                    for value in values:
                        total_sum += float(value[1])  
                        value_count += 1  

                gpu_eff = (total_sum / value_count) if value_count > 0 else 0
                with open('/var/www/max-reports/ReportingTool/gpu_requests.log', 'a') as log_file:
                    log_file.write(f"JobID: {jobID}\n")
                    log_file.write(f"Request URL: {response.url}\n")
                    log_file.write(f"Response: {response.text}\n")
                    log_file.write("\n")

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            print(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"An error occurred: {req_err}")
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    try:
        update_query = """
            UPDATE slurm
            SET NGpus = ?, GpuUtil = ?, ReqGPUS = ?
            WHERE JobID = ?  
        """
        cur.execute(update_query, (total_gpus, gpu_eff, gpu_available, jobID))
        con.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    con = sqlite3.connect('max-reports-slurm.sqlite3')
    get_rows_without_gpu(con)
