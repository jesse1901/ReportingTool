def get_config():
    config = {
    "fields":  {
        "jobID": {
            "label": "Job ID",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "username": {
            "label": "Username",
            "type": "text",
            "operators": ["=", "!="]
        },
        "account": {
            "label": "Account",
            "type": "text",
            "operators": ["=", "!="]
        },
        "cpu_efficiency": {
            "label": "CPU Efficiency",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "lost_cpu_time": {
            "label": "Lost CPU Time",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "lost_cpu_time_sec": {
            "label": "Lost CPU Time (sec)",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "gpu_efficiency": {
            "label": "GPU Efficiency",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "lost_gpu_time": {
            "label": "Lost GPU Time",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "lost_gpu_time_sec": {
            "label": "Lost GPU Time (sec)",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "real_time": {
            "label": "Real Time",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "job_cpu_time": {
            "label": "Job CPU Time",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "real_time_sec": {
            "label": "Real Time (sec)",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "state": {
            "label": "State",
            "type": "text",
            "operators": ["=", "!="]
        },
        "cores": {
            "label": "Cores",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "gpu_nodes": {
            "label": "GPU Nodes",
            "type": "number",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "start": {
            "label": "Start Time",
            "type": "datetime",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "end": {
            "label": "End Time",
            "type": "datetime",
            "operators": ["=", ">", "<", ">=", "<="]
        },
        "job_name": {
            "label": "Job Name",
            "type": "text",
            "operators": ["=", "!="]
        },
        "partition": {
            "label": "Partition",
            "type": "text",
            "operators": ["=", "!="]
        }
    }
    }
    return config
