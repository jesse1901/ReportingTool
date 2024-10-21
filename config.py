def get_config():
    return {
        "fields": {
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
            "lost_gpu_time_sec": {
                "label": "Lost GPU Time (sec)",
                "type": "number",
                "operators": ["=", ">", "<", ">=", "<="]
            },
            "state": {
                "label": "State",
                "type": "text",
                "operators": ["=", "!="]
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
            # Add more fields here as needed
        }
    }