# Max-Reports Reporting Tool

This repository contains the necessary scripts and configuration to run the **Max-Reports** application, which processes SLURM data and displays results via a Streamlit web application. It includes services to run background tasks such as data collection and periodic execution of scripts, and timers to manage regular updates.

## Prerequisites

Before setting up the services and running the application, ensure you have the following installed:

- Python 3.8+
- Systemd (for managing services and timers)
- Virtual environment (`python3-venv`)
- SLURM and related tools (for the SLURM data processing)

## Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jesse1901/ReportingTool.git
   cd ReportingTool
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. **Install the required Python dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install slurm2sql
    ```

4. **Set up Streamlit configuration files:**

    Create a `.streamlit` directory in your project root and set up the `secrets.toml` and `config.toml` files.

    ```bash
    mkdir -p .streamlit
    nano .streamlit/config.toml
    ```

    Add the following content to `config.toml`:

    ```toml
    # General Settings
    [server]
    headless = true
    port = 8501
    sslCertFile = ''
    sslKeyFile = ''

    # Logging Settings
    [logger]
    level = "warning"

    # Client Side Settings
    [client]
    showErrorDetails = false
    toolbarMode = "viewer"

    [browser]
    gatherUsageStats = false

    [theme]
    base = "dark"


    ```

    ```bash
    nano .streamlit/secrets.toml
    ```

    Add the following content to `secrets.toml`:

    ```toml
    [ldap]
    server_path = ""
    domain = ""
    search_base = ""
    attributes = ["sAMAccountName", "distinguishedName", "userPrincipalName", "displayName", "manager", "title"]
    use_ssl = true

    [session_state_names]
    user = "login_user"
    remember_me = "login_remember_me"

    [auth_cookie]
    name = "login_cookie"
    key = "{any password for encryption}"
    expiry_days = 1
    auto_renewal = true
    delay_sec = 0.1

    [users]
    allowed_users = ['user']
    admin_users = ['admin']
    xfel_users = ['xefl_user']
    ```

5. **Set up systemd services and timers:**

   Create the necessary systemd service and timer files to manage the periodic execution of SLURM data extraction, GPU data gathering, and running the Streamlit application.

   - **slurm2sql.service:**

     This service will run the `slurm2sql` command to process SLURM data.

     ```bash
     sudo nano /etc/systemd/system/slurm2sql.service
     ```

     ```ini
     [Unit]
     Description=Run slurm2sql hourly
     After=network.target

     [Service]
     Type=oneshot
     ExecStart=/usr/local/bin/slurm2sql --history-resume /path/to/your/database.sqlite3 -- -a

     [Install]
     WantedBy=multi-user.target
     ```

   - **slurm2sql.timer:**

     This timer will trigger the `slurm2sql.service` every hour.

     ```bash
     sudo nano /etc/systemd/system/slurm2sql.timer
     ```

     ```ini
     [Unit]
     Description=Timer to run slurm2sql every hour

     [Timer]
     OnCalendar=hourly
     Persistent=true

     [Install]
     WantedBy=timers.target
     ```

   - **get_gpu_data_prometheus.service:**

     This service will run the `get_gpu.py` script to gather GPU data.

     ```bash
     sudo nano /etc/systemd/system/get_gpu_data_prometheus.service
     ```

     ```ini
     [Unit]
     Description=Run get_gpu.py script
     After=slurm2sql.service
     Requires=slurm2sql.service

     [Service]
     Type=oneshot
     ExecStart=/path/to/your/virtualenv/bin/python /path/to/your/project/get_gpu.py
     WorkingDirectory=/path/to/your/project/
     Environment="PATH=/path/to/your/virtualenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
     Environment="LD_LIBRARY_PATH=/usr/local/lib:/usr/lib"
     StandardOutput=journal
     StandardError=journal

     [Install]
     WantedBy=multi-user.target
     ```

   - **get_gpu_data_prometheus.timer:**

     This timer will trigger the `get_gpu_data_prometheus.service` every hour.

     ```bash
     sudo nano /etc/systemd/system/get_gpu_data_prometheus.timer
     ```

     ```ini
     [Unit]
     Description=Run get_gpu.py script every hour

     [Timer]
     OnCalendar=hourly
     Persistent=true

     [Install]
     WantedBy=timers.target
     ```

   - **max-reports.service:**

     This service runs the Streamlit application to display the reports via the web interface.

     ```bash
     sudo nano /etc/systemd/system/max-reports.service
     ```

     ```ini
     [Unit]
     Description=Streamlit Max-Reports Application
     After=network.target

     [Service]
     ExecStart=/path/to/your/virtualenv/bin/python -m streamlit run /path/to/your/project/app.py --server.port=8501 --server.headless=true
     WorkingDirectory=/path/to/your/project/
     Environment="PATH=/path/to/your/virtualenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
     Environment="LD_LIBRARY_PATH=/usr/local/lib:/usr/lib"
     StandardOutput=journal
     StandardError=journal
     Environment=PYTHONUNBUFFERED=1

     [Install]
     WantedBy=multi-user.target
     ```

5. **Enable and start the services and timers:**

   After creating the service and timer files, enable and start the services and timers with the following commands:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable slurm2sql.service slurm2sql.timer
   sudo systemctl enable get_gpu_data_prometheus.service get_gpu_data_prometheus.timer
   sudo systemctl enable max-reports.service

   sudo systemctl start slurm2sql.service slurm2sql.timer
   sudo systemctl start get_gpu_data_prometheus.service get_gpu_data_prometheus.timer
   sudo systemctl start max-reports.service
   ```

6. **Access the Max-Reports Streamlit Application:**

   The Streamlit application will be accessible at `http://localhost:8501`. You can access it through a web browser.
   To make the Streamlit application accessible through a domain, you can set up NGINX as a reverse proxy or edit the NGINX configuration to route traffic to http://localhost:8501.

## Customization

- **SLURM Data Location:** Make sure to update the paths to your SLURM database and relevant Python scripts in the service files (e.g., `/path/to/your/database.sqlite3`, `/path/to/your/project/`).
- **Port Configuration:** If you need to change the port for the Streamlit application, modify the `--server.po rt` argument in the `max-reports.service` file.

## Troubleshooting

- **Check service status:**
  ```bash
  sudo systemctl status <service-name>
  ```
- **View logs for debugging:**
  ```bash
  journalctl -u <service-name>
  ```

