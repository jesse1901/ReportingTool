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
from figures import CreateFigures


secrets = toml.load('.streamlit/secrets.toml')

# LDAP Configuration
LDAP_SERVER = secrets['ldap']['server_path']
SEARCH_BASE = secrets['ldap']['search_base']
USE_SSL = secrets['ldap']['use_ssl']

ALLOWED_USERS = secrets['users']['allowed_users']
ADMIN_USERS = secrets['users']['admin_users']

def authenticate(username, password):
    if not password:
        st.error("Password cannot be empty")
        return False

    try:
        # Create an LDAP server object
        server = Server(LDAP_SERVER, use_ssl=USE_SSL, get_info=ALL)

        # Adjust this user format according to your LDAP server requirements
        user = f"uid={username},ou=people,ou=rgy,o=desy,c=de"

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

def is_user_admin(username):
    return username in ADMIN_USERS

def main():
    if 'admin' in st.session_state:
        username = st.session_state['admin']
        tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data", "Efficiency", "Total"])

        with tab1:
            st.header("User Data")
            col1, col2 = st.columns([3, 1])
            with col1:
                create.frame_user_all(username)
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

    elif 'user' in st.session_state:
        username = st.session_state['user']
    
        create.frame_user_all(username)
    else:
        st.title("Login Max-Reports")
        form = st.form(key="login_form")
        # Show login form if user is not authenticated
        username = form.text_input("Username")
        password = form.text_input("Password", type="password")
        try:
            if form.form_submit_button("Login"):
                if authenticate(username, password):
                    if is_user_admin(username):
                        st.session_state['admin'] = username
                        st.success('success')
                        st.rerun()
                    elif is_user_allowed(username):        
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



