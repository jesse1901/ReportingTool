import requests
import sqlite3
import hostlist
import toml
import gpu_node_data
from collections import defaultdict


def get_available_gpus_per_node(prom_base_url):
    url = f"{prom_base_url}/api/v1/query?query=nvidia_smi_gpu_info"

    r = requests.get(url)
    r = r.json()

    gpu_d = r["data"]["result"]

    gpu_count = defaultdict(int)
    list = []

    for i in gpu_d:
        instance = i["metric"]["instance"]
        gpu_count[instance] += 1

    for node, count in gpu_count.items():
        list.append((node , count))

    # Ergebnisse speichern
    with open("gpu_node_data.py", "w") as f:
        f.write(f"""def hostlist_gpu(): 
                              gpu_nodes = {list}
                              return gpu_nodes""")

    print("GPU counts saved to gpu_node_data.py.txt")


def get_rows_without_gpu(con, prom_base_url):
    cur = con.cursor()
    step = 1
    jobs_processed = 0  # Counter for successfully processed jobs

    while True:
        try:
            cur.execute("""SELECT Start, End, NodeList, Elapsed, JobID 
                FROM allocations 
                    WHERE 
                (NGPUS > 0 AND GpuUtil IS NULL AND Start IS NOT NULL AND End IS NOT NULL AND State != 'RUNNING') 
                     OR 
                (ReqGPUS IS NULL AND Start IS NOT NULL AND End IS NOT NULL AND State != 'RUNNING')""")
            
            data = cur.fetchall()
            if not data:
                if jobs_processed > 0:
                    print(f"{jobs_processed} jobs were successfully updated with GPU data.")
                else:
                    print("No jobs to process.")
                break

            for row in data:
                if row[0] and row[1]:  # Ensure Start and End are not None
                    try:
                        get_gpu_data(cur, row, step, prom_base_url=prom_base_url)
                        jobs_processed += 1  # Increment the processed jobs counter
                    except Exception as e:
                        print(f"Failed to process JobID {row[4]}: {e}")

            step += 1
            if step > 10:
                print(f"No response for {len(data)} jobs remaining.")
                break

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break


def get_gpu_data(cur, row, step, prom_base_url):
    start = row[0]
    end = row[1]
    nodelist = row[2]
    elapsed = row[3]
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
        prometheus_url = f'{prom_base_url}/api/v1/query_range'
        params = {
            'query': f'nvidia_smi_utilization_gpu_ratio{{instance=~"{join_nodes}"}}',
            'start': f'{start}',
            'end': f'{end}',
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

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")

    try:
        update_query = """
            UPDATE slurm
            SET NGpus = ?, GpuUtil = ?, ReqGPUS = ?
            WHERE JobID = ?  
        """
        cur.execute(update_query, (total_gpus, gpu_eff, gpu_available, jobID))
        con.commit()
    except sqlite3.Error as e:
        print(f"SQLite error during update: {e}")


if __name__ == "__main__":
    secrets = toml.load('.streamlit/secrets.toml')

    prom_base_url = secrets['urls']['prometheus']

    con = sqlite3.connect('/var/www/max-reports/ReportingTool/max-reports-slurm.sqlite3')
    get_available_gpus_per_node(prom_base_url)
    get_rows_without_gpu(con, prom_base_url)
