def get_config():
    config = {
        "fields": {
            "jobID": {
                "label": "Job ID",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "username": {
                "label": "Username",
                "type": "text",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "like", "not_like"]
            },
            "account": {
                "label": "Account",
                "type": "text",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "like", "not_like"]
            },
            "cpu_efficiency": {
                "label": "CPU Efficiency",
                "type": "number",
                "fieldSettings": {
                    "min": 0,
                    "max": 100
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "lost_cpu_time": {
                "label": "Lost CPU Time",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "lost_cpu_time_sec": {
                "label": "Lost CPU Time (sec)",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "gpu_efficiency": {
                "label": "GPU Efficiency",
                "type": "number",
                "fieldSettings": {
                    "min": 0,
                    "max": 100
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "lost_gpu_time": {
                "label": "Lost GPU Time",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "lost_gpu_time_sec": {
                "label": "Lost GPU Time (sec)",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "real_time": {
                "label": "Real Time",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "job_cpu_time": {
                "label": "Job CPU Time",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "real_time_sec": {
                "label": "Real Time (sec)",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "state": {
                "label": "State",
                "type": "select",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal"],
                "fieldSettings": {
                    "listValues": [
                        {"value": "running", "title": "Running"},
                        {"value": "completed", "title": "Completed"},
                        {"value": "failed", "title": "Failed"}
                    ]
                }
            },
            "cores": {
                "label": "Cores",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "gpu_nodes": {
                "label": "GPU Nodes",
                "type": "number",
                "fieldSettings": {
                    "min": 0
                },
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "start": {
                "label": "Start Time",
                "type": "datetime",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "end": {
                "label": "End Time",
                "type": "datetime",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "less", "less_or_equal", "greater", "greater_or_equal"]
            },
            "job_name": {
                "label": "Job Name",
                "type": "text",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "like", "not_like"]
            },
            "partition": {
                "label": "Partition",
                "type": "text",
                "valueSources": ["value"],
                "operators": ["equal", "not_equal", "like", "not_like"]
            }
        }
    }

    return config
