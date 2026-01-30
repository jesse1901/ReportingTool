import requests
import duckdb
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
    list_nodes = []

    for i in gpu_d:
        instance = i["metric"]["instance"]
        gpu_count[instance] += 1

    for node, count in gpu_count.items():
        list_nodes.append((node, count))

    with open("gpu_node_data.py", "w") as f:
        f.write(f"""def hostlist_gpu(): 
                              gpu_nodes = {list_nodes}
                              return gpu_nodes""")

    print("GPU counts saved to gpu_node_data.py.txt")


def bulk_update_duckdb(con, updates):
    """
    Performs a bulk update using a temporary table.
    'updates' is a list of tuples: (NGpus, GpuUtil, ReqGPUS, JobID)
    """
    if not updates:
        return

    cur = con.cursor()
    try:
        cur.execute("""
            CREATE TEMPORARY TABLE IF NOT EXISTS gpu_update_stage (
                NGpus INTEGER, 
                GpuUtil DOUBLE, 
                ReqGPUS INTEGER, 
                JobID VARCHAR
            )
        """)
        
        # Clear any old data
        cur.execute("DELETE FROM gpu_update_stage")

        # 2. Insert all python data into the temp table in one fast operation
        cur.executemany(
            "INSERT INTO gpu_update_stage VALUES (?, ?, ?, ?)", 
            updates
        )

        # 3. Perform a single JOIN UPDATE to update the main 'slurm' table
        cur.execute("""
            UPDATE slurm
            SET 
                NGpus = stage.NGpus,
                GpuUtil = stage.GpuUtil,
                ReqGPUS = stage.ReqGPUS
            FROM gpu_update_stage AS stage
            WHERE slurm.JobID = stage.JobID
        """)

        # 4. Clean up
        cur.execute("DROP TABLE gpu_update_stage")
        con.commit()
        
    except duckdb.Error as e:
        print(f"DuckDB Batch Update Error: {e}")


def get_rows_without_gpu(con, prom_base_url):
    cur = con.cursor()
    step = 1
    jobs_processed = 0

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

            # Buffer to hold updates for this batch
            batch_updates = []

            for row in data:
                if row[0] and row[1]:  # Ensure Start and End are not None
                    try:

                        result = get_gpu_data(row, step, prom_base_url=prom_base_url)
                        
                        if result:
                            # result is (total_gpus, gpu_eff, gpu_available, jobID)
                            batch_updates.append(result)
                            jobs_processed += 1

                    except Exception as e:
                        print(f"Failed to process JobID {row[4]}: {e}")

            # Perform the BULK UPDATE for all successful calculations in this step
            if batch_updates:
                bulk_update_duckdb(con, batch_updates)

            step += 1
            if step > 10:
                print(f"No response for {len(data)} jobs remaining.")
                break

        except duckdb.Error as e:
            print(f"DuckDB error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break


def get_gpu_data(row, step, prom_base_url):
    """
    Calculates GPU usage and returns a tuple (NGpus, GpuUtil, ReqGPUS, JobID).
    Returns None if no GPU nodes were found.
    """
    start = row[0]
    end = row[1]
    nodelist = row[2]
    # elapsed = row[3] # Unused
    jobID = row[4]
    
    nodelist = hostlist.expand_hostlist(nodelist)
    nodelist = [host + '.desy.de' for host in nodelist]

    all_gpu_nodes = gpu_node_data.hostlist_gpu()

    # Determine which nodes in the job are actually GPU nodes
    gpu_nodes = [node for node in nodelist if any(node == gpu_node[0] for gpu_node in all_gpu_nodes)]
    total_gpus = sum(gpu_count for node, gpu_count in all_gpu_nodes if node in gpu_nodes)

    gpu_eff = None

    if gpu_nodes:
        gpu_available = total_gpus
    else:
        gpu_available = None

    # Only query prometheus if we actually have GPU nodes
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
            print(f"Request error for JobID {jobID}: {e}")
            return None # Skip update for this row on error

    # Return the data to be inserted
    return (total_gpus, gpu_eff, gpu_available, jobID)


if __name__ == "__main__":
    secrets = toml.load('.streamlit/secrets.toml')

    prom_base_url = secrets['urls']['prometheus']

    con = duckdb.connect('/var/www/max-reports/ReportingTool/max-reports_v1.duckdb')
    
    get_available_gpus_per_node(prom_base_url)
    get_rows_without_gpu(con, prom_base_url)