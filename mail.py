import toml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import calendar
import sqlite3
from ldap3 import Server, Connection, ALL

# Load secrets from TOML file
secrets = toml.load('.streamlit/secrets.toml')
smtp_server = "smtp.desy.de"
smtp_port = 25
from_email = "maxwell.service@desy.de"

# Get list of allowed users (emails)
#ALLOWED_USERS = secrets['users']['allowed_users']

LDAP_SERVER = 'ldap://ldap.desy.de'
SEARCH_BASE = "ou=people,ou=rgy,o=desy,c=de"
USE_SSL = False

def get_month_range():
    month = datetime.now().month
    year = datetime.now().year
    # First day of the month

    # Last day of the month (by going to the first day of the next month and subtracting one day)
    if month == 1:
        last_month = 12
        first_day = datetime(year - 1, last_month , 1)
        last_day = datetime(year, 1, 1) - timedelta(days=1)
    else:
        last_month = month - 1
        first_day = datetime(year, month - 1, 1)
        last_day = datetime(year, month, 1) - timedelta(days=1)


    first_day_unix = int(first_day.timestamp())
    last_day_unix = int(last_day.timestamp())
    return first_day_unix, last_day_unix, last_month, year


def get_user_mail_ldap(username):
    try:
        # Create the LDAP server object
        server = Server(LDAP_SERVER, get_info=ALL)

        # Establish an anonymous connection
        conn = Connection(server, auto_bind=True)

        # Perform the search
        search_filter = f"(uid={username})"
        conn.search(
            search_base=SEARCH_BASE,
            search_filter=search_filter,
            attributes=["mail"]  # Request only the 'mail' attribute
        )

        if conn.entries:
            # Retrieve the mail attribute value
            email = conn.entries[0].mail.value
            return email
        else:
            print(f"User {username} not found or no email attribute available.")
            return None
    except Exception as e:
        print(f"LDAP search error: {e}")
        return None


    

def send_email(to_email, subject, body):
    # create MIME object
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach body text
    msg.attach(MIMEText(body, 'html'))

    # Connect to the mail server and send message
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        #server.starttls()               # Upgrade the connection to a secure TLS/SSL mode
        server.sendmail(from_email, to_email, msg.as_string())

def get_sql_data(user, first_day_unix, last_day_unix):
    con = sqlite3.connect('max-reports-slurm.sqlite3')
    cur = con.cursor()
    userx = 'akorol'
    cur.execute("""                    
            SELECT
            IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category, COUNT(JobID) AS JobCount
            FROM allocations
            WHERE Partition != 'jhub' AND State IS NOT 'PENDING' AND State IS NOT 'RUNNING' 
            AND Start >= ? AND End <= ? AND JobName != 'interactive' AND User = ? GROUP BY Category""",
            (first_day_unix, last_day_unix, userx))
    states = cur.fetchall()

    cur.execute(""" 
    SELECT
    eff.User,
    eff.Account,
    COUNT(eff.JobID) AS JobCount,
    ROUND(
        CASE 
            WHEN SUM((eff.cpu_s_reserved / 2) - eff.cpu_s_used) / 86400 < 0 THEN 0
            ELSE SUM((eff.cpu_s_reserved / 2) - eff.cpu_s_used) / 86400
        END, 
        1
    ) AS Lost_CPU_days,

    ROUND(SUM(slurm.CPUTime) / 2 / 86400, 1) AS cpu_days,
    ROUND(iif((100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) * 2 > 100, 100, (100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) * 2), 1) AS CPUEff,

    ROUND(SUM(eff.Elapsed * eff.NGPUs) / 86400, 1) AS GPU_Days,
    ROUND(SUM((eff.NGPUS * eff.Elapsed) * (1 - eff.GPUeff)) / 86400, 1) AS Lost_GPU_Days,
    ROUND(
        iif(
            SUM(eff.NGPUs), 
            100 * SUM(eff.Elapsed * eff.NGPUs * eff.GPUeff) / SUM(eff.Elapsed * eff.NGPUs),
            NULL
        ),
        1
    ) AS GPUEff,
    ROUND(SUM(eff.TotDiskRead / 1048576) / SUM(eff.Elapsed), 2) AS read_MiBps,
    ROUND(SUM(eff.TotDiskWrite / 1048576) / SUM(eff.Elapsed), 2) AS write_MiBps
  FROM eff
  JOIN slurm ON eff.JobID = slurm.JobID
  WHERE eff.Start >= ? 
  AND eff.End <= ? 
  AND eff.End IS NOT NULL 
  AND slurm.Partition != 'jhub'
  AND slurm.JobName != 'interactive'
  AND eff.User = ?
  GROUP BY eff.User
                """, (first_day_unix, last_day_unix, userx)) 
    rows = cur.fetchall()

    for row in rows:
        total_job_count = row[2]
        total_lost_cpu_days = row[3]
        total_cpu_days = row[4]
        cpu_eff = row[5]
        total_gpu_days = row[6]
        total_lost_gpu_days = row[7]
        gpu_eff = row[8]

    state_dict = {state[0]: state[1] for state in states}
    return state_dict, total_job_count, total_lost_cpu_days, total_cpu_days, total_gpu_days, total_lost_gpu_days, cpu_eff, gpu_eff

def get_sql_data_by_partition(first_day_unix, last_day_unix, partition):
    con = sqlite3.connect('max-reports-slurm.sqlite3')
    cur = con.cursor()

    # Query for job states by partition
    cur.execute("""
        SELECT
        Partition,
        IIF(LOWER(State) LIKE 'cancelled %', 'CANCELLED', State) AS Category,
        COUNT(JobID) AS JobCount
        FROM allocations
        WHERE Partition != 'jhub'
          AND State IS NOT 'PENDING'
          AND State IS NOT 'RUNNING'
          AND Start >= ? 
          AND End <= ? 
          AND JobName != 'interactive'
          AND Partition = ?
        GROUP BY Category
    """, (first_day_unix, last_day_unix, partition))
    state_data = cur.fetchall()
    # Query for partition-based statistics
    cur.execute("""
        SELECT
        slurm.Partition,
        COUNT(eff.JobID) AS JobCount,
        ROUND(
            CASE 
                WHEN SUM((eff.cpu_s_reserved / 2) - eff.cpu_s_used) / 86400 < 0 THEN 0
                ELSE SUM((eff.cpu_s_reserved / 2) - eff.cpu_s_used) / 86400
            END, 
            1
        ) AS Lost_CPU_days,
            
        ROUND(SUM(slurm.CPUTime) / 2 / 86400, 1) AS cpu_days,
        
                
        ROUND(
            IIF(
                (100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) * 2 > 100,
                100,
                (100 * SUM(eff.Elapsed * eff.NCPUS * eff.CPUEff) / SUM(eff.Elapsed * eff.NCPUS)) * 2
            ),
            1
        ) AS CPUEff,
        ROUND(SUM(eff.Elapsed * eff.NGPUs) / 86400, 1) AS GPU_Days,
        ROUND(SUM((eff.NGPUs * eff.Elapsed) * (1 - eff.GPUeff)) / 86400, 1) AS Lost_GPU_Days,
        ROUND(
            IIF(
                SUM(eff.NGPUs),
                100 * SUM(eff.Elapsed * eff.NGPUs * eff.GPUeff) / SUM(eff.Elapsed * eff.NGPUs),
                NULL
            ),
            1
        ) AS GPUEff
        FROM eff
        JOIN slurm ON eff.JobID = slurm.JobID
        WHERE eff.Start >= ? 
          AND eff.End <= ? 
          AND eff.End IS NOT NULL 
          AND slurm.Partition != 'jhub'
          AND slurm.JobName != 'interactive'
          AND slurm.partition = ? 
        GROUP BY slurm.Partition
    """, (first_day_unix, last_day_unix, partition))
    partition_stats = cur.fetchall()

    # Process job states

    state_dict = {row[1]: row[2] for row in state_data}
    partition_results = {}
    for row in partition_stats:
        partition = row[0]
        partition_results[partition] = {
            "JobCount": row[1],
            "Lost_CPU_Days": row[2],
            "CPU_Days": row[3],
            "CPU_Efficiency": row[4],
            "GPU_Days": row[5],
            "Lost_GPU_Days": row[6],
            "GPU_Efficiency": row[7]
        }

    con.close()
    return state_dict, partition_results





if __name__ == "__main__":
  first_day_unix, last_day_unix, last_month, year = get_month_range()
  last_month_name = calendar.month_name[last_month]

  PARTITION_ADMINS = {"exfel":"schuetzj"}
  ALLOWED_USERS = ["schuetzj"]

  # Iterate over all allowed users and send them an email
  for user in ALLOWED_USERS:
      try:
          state_dict, total_job_count, total_lost_cpu_days, total_cpu_days, total_gpu_days, total_lost_gpu_days, cpu_eff, gpu_eff = get_sql_data(user, first_day_unix, last_day_unix)
          user_email = get_user_mail_ldap(user)
          subject = f"Maxwell efficiency Report for {last_month_name} {year}"
          if total_job_count == 0:
              print(f"No jobs found for {user}")
              continue
          if cpu_eff < 25:
            usage_label = "Bad"
            usage_color = "red"
          elif cpu_eff < 50:
            usage_label = "Medium"
            usage_color = "orange"
          else:
            usage_label = "Good"
            usage_color = "green"
          body = f"""
  <html>
    <head></head>
    <body style="font-family: Arial, sans-serif;">

      <p>Hello {user},</p>

      <p>
        We hope you’re doing well! Below is a detailed overview of your Maxwell HPC usage for {last_month_name} {year}. 
        Over the last month, your utilization level was
        <strong style="color: {usage_color};">
          {usage_label}
        </strong>
      </p>

      <ul>
          <li>
            <strong>Total Jobs Run:</strong> {total_job_count}
            <ul>
              {''.join(f"<li>{state}: {count}</li>" for state, count in state_dict.items())}
            </ul>
        </li>
          <li><strong>Total CPU Time (Days):</strong> {total_cpu_days}</li>
          <li><strong>Lost CPU Time (Days):</strong> {total_lost_cpu_days}</li>
          <li><strong>CPU Efficiency:</strong> {cpu_eff}%</li>
          <li><strong>Total GPU Time (Days):</strong> {total_gpu_days}</li>
          <li><strong>Lost GPU Time (Days):</strong> {total_lost_gpu_days}</li>
          <li><strong>GPU Efficiency:</strong> {gpu_eff}%</li>
       </ul>

      <p>
        We hope this information helps you understand and optimize your workload on Maxwell. 
        For futher details on these statistics, please visit <a href="https://docs.desy.de">docs.desy.de</a>.
      </p>

      <p>
      If you have any questions or need assistance, please feel free to reach out.
      Thank you for using Maxwell, and have a productive month ahead!</p>

      <p>
        Best regards,<br />
        <em>Your Maxwell Team</em>
      </p>

    </body>
  </html>
  """
          send_email(user_email, subject, body)
          print(f"Successfully sent email to {user_email}")
      except Exception as e:
          print(f"Failed to send email to {user}: {e}")   




  for partition, admin in PARTITION_ADMINS.items():
        try:
            state_dict, partition_results = get_sql_data_by_partition(first_day_unix, last_day_unix, partition)
            admin_email = get_user_mail_ldap(admin)

            stats = partition_results[partition]
            subject = f"Maxwell Partition Report for {partition} ({last_month_name} {year})"
            body = f"""
            <html>
              <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <p>Hello,</p>
                <p>
                  We hope you’re doing well! Below is a detailed overview of the Maxwell HPC usage 
                  for the <strong>{partition}</strong> partition during 
                  <strong>{last_month_name} {year}</strong>.
                </p>

                <ul>
                  <li>
                    <strong>Total Jobs Run:</strong> {stats['JobCount']}
                    <ul>
                      {''.join(f"<li>{state}: {count}</li>" for state, count in state_dict.items())}
                    </ul>
                  </li>
                  <li><strong>Total CPU Time (Days):</strong> {stats['CPU_Days']}</li>
                  <li><strong>Lost CPU Time (Days):</strong> {stats['Lost_CPU_Days']}</li>
                  <li><strong>CPU Efficiency:</strong> {stats['CPU_Efficiency']}%</li>
                  <li><strong>Total GPU Time (Days):</strong> {stats['GPU_Days']}</li>
                  <li><strong>Lost GPU Time (Days):</strong> {stats['Lost_GPU_Days']}</li>
                  <li><strong>GPU Efficiency:</strong> {stats['GPU_Efficiency']}%</li>
                </ul>

                <p>
                  For additional insights, please visit 
                  <a href="https://max-portal.desy.de/reporting">max-portal.desy.de/reporting</a>. 
                  If you have any questions or require further assistance, feel free to reach out. 
                </p>
                <p>
                  Best regards,<br/>
                  <em>Your Maxwell HPC Team</em>
                </p>
              </body>
            </html>
            """


            send_email(admin_email, subject, body)
            print(f"Successfully sent email to partition admin: {admin_email}")

        except Exception as e:
            print(f"Failed to send email to admin for partition {partition}: {e}")
    

