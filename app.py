import pyslurm
import streamlit as st
import pandas as pd
import time
from config import get_config
from datetime import timedelta, datetime
import numpy as np
import sqlite3
import toml
from ldap3 import Server, Connection, ALL
from streamlit_condition_tree import condition_tree, config_from_dataframe
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from dataclasses import asdict


color_map = {
    'CANCELLED': '#1f77b4 ',    # Light Blue
    'COMPLETED': '#17becf ',    # Light Sky Blue
    'TIMEOUT': '#d62728 ',     # red
    'FAILED': '#e377c2',      # Pink
    'PREEMPTED': '#2ca02c',     # Light Green
    'NODE_FAIL': '#fcf76a'
}
secrets = toml.load('.streamlit/secrets.toml')

# LDAP Configuration
LDAP_SERVER = secrets['ldap']['server_path']
SEARCH_BASE = secrets['ldap']['search_base']
USE_SSL = secrets['ldap']['use_ssl']

ALLOWED_USERS = secrets['users']['allowed_users']


def authenticate(username, password):
    if not password:
        st.error("Password cannot be empty")
        return False

    try:
        # Create an LDAP server object
        server = Server(LDAP_SERVER, use_ssl=USE_SSL, get_info=ALL)

        # Adjust this user format according to your LDAP server requirements
        user = f"uid={username},ou=people,ou=rgy,o=desy,c=de"  # Change if necessary

        # Establish a connection
        conn = Connection(server, user=user, password=password.strip())  # Ensure no whitespace in password
        
        if conn.bind():
            return True  # Authentication successful
        else:
            st.error("Invalid username or password")  # Provide feedback for failed login
            return False  # Authentication failed
    except Exception as e:
        st.error(f"LDAP connection error: {e}")
        return False

def is_user_allowed(username):
    return username in ALLOWED_USERS


def timestring_to_seconds(timestring):
    if pd.isna(timestring) or timestring == '0' or timestring == 0 or timestring.strip() == '':
        return 0

    if isinstance(timestring, float):
        timestring = str(int(timestring))  # Convert float to integer string

    # Ensure timestring is in string format
    timestring = str(timestring).strip()

    # Split by "T" to separate days from time
    if 'T' in timestring:
        days_part, time_part = timestring.split('T')
    else:
        days_part, time_part = '0', timestring

    # Convert days part
    try:
        days = int(days_part.strip()) if days_part.strip() else 0
    except ValueError:
        days = 0

    # Convert time part (HH:MM:SS)
    time_parts = time_part.split(':')
    try:
        hours = int(time_parts[0].strip()) if len(time_parts) > 0 and time_parts[0].strip() else 0
    except ValueError:
        hours = 0
    try:
        minutes = int(time_parts[1].strip()) if len(time_parts) > 1 and time_parts[1].strip() else 0
    except ValueError:
        minutes = 0
    try:
        seconds = int(time_parts[2].strip()) if len(time_parts) > 2 and time_parts[2].strip() else 0
    except ValueError:
        seconds = 0

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


def format_interval_label(interval):
    min_time = interval.left
    max_time = interval.right

    def format_time(minutes):
        days = int(minutes // 1440)  # 1440 minutes in a day
        hours = int((minutes % 1440) // 60)
        mins = int(minutes % 60)

        if days > 0 and hours > 0:
            return f"{days}d {hours}h"
        elif days > 0:
            return f"{days}d"
        elif hours > 0 and mins > 0:
            return f"{hours}h {mins}m"
        elif hours > 0:
            return f"{hours}h"
        else:
            return f"{mins}m"

    min_time_str = format_time(min_time)
    max_time_str = format_time(max_time)
    return f"{min_time_str} - {max_time_str}"



def main():
        
    if 'user' in st.session_state:    
        # Tabs erstellen
        tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data", "Efficiency", "Total"])

        with tab1:
            st.header("User Data")
            col1, col2 = st.columns([3, 1])
            with col1:
                create.frame_user_all()
            with col2:
                create.frame_group_by_user()

        with tab2:
            st.header("Job Data")
            col3, col4, col5 = st.columns(3)
            with col3:
                create.job_counts_by_log2()
            with col4:
                create.pie_chart_job_count()
            with col5:
                create.pie_chart_batch_inter()

        with tab3:
            st.header("Efficiency")
            col6, col7, col8 = st.columns(3)
            with col6:
                create.pie_chart_by_session_state()
            with col7:
                create.pie_chart_by_job_count()
            with col8:
                create.efficiency_percentile_chart()
            # create.chart_cpu_utilization()

        with tab4:
            st.header("")
            col9, col10 = st.columns(2)
            with col9:
                create.bar_char_by_user()
            with col10:
                create.scatter_chart_data_cpu_gpu_eff()
    else:
        st.title("Login Max-Reports")
        form = st.form(key="login_form")
        # Show login form if user is not authenticated
        username = form.text_input("Username")
        password = form.text_input("Password", type="password")
        try:
            if form.form_submit_button("Login"):
                if authenticate(username, password):
                    if is_user_allowed(username):        
                        st.session_state['user'] = username
                        st.success('success')
                        st.rerun()
                    
                    else:
                        st.error("you are not authorized to login")    

        except Exception as e:
            st.error("")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    con = sqlite3.connect('reports.db')
    create = CreateFigures(con)
    st_autorefresh(interval=600000)
    main()



