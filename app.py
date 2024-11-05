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
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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

def date_slider_wrapper(func, current_user, role, key:str, number_input=False, hyper_threading=False):
# Wrapper to handle Date-Selection with st.cache
       
    default_range = [datetime.today() - timedelta(days=30), datetime.today()]
    date_selector = st.date_input("select timerange", default_range, key=f"date_slider_{key}")
    
    if len(date_selector) != 2:
        st.stop()

    start_date, end_date = date_selector

    if hyper_threading:
        scale_efficiency = st.checkbox("Hyperthreading", key=f"checkbox_{key}")

        if 'scale_efficiency' not in st.session_state:
            st.session_state.scale_efficiency = False 
    

    if number_input and hyper_threading:
        display_user = st.number_input("Number of Users:", value=20, key=f"number_input_{key}" )
        func( start_date, end_date, current_user, role, scale_efficiency, display_user)
    
    elif number_input: 
        display_user = st.number_input("Number of Users:", value=20, key=f"number_input_{key}")
        func( start_date, end_date, current_user, role, display_user)
    
    elif hyper_threading: 
        st.markdown("<div style='visibility:hidden;height:42px;'></div>", unsafe_allow_html=True)
        func( start_date, end_date, current_user, role, hyper_threading)
    
    else: 
        func( start_date, end_date, current_user, role)


def is_user_allowed(username):
    return username in ALLOWED_USERS

def is_user_admin(username):
    return username in ADMIN_USERS


def main():
    if 'user_role' in st.session_state:
        username = st.session_state['username']
        user_role = st.session_state['user_role']
        
        if user_role == 'admin':
            tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data", "Efficiency", "Total"])

            with tab1:
                st.header("User Data")
                col1, col2 = st.columns([3, 1])
                with col1:
                    create.frame_user_all(username, user_role)
                    create.get_job_script()
                with col2:
                    date_slider_wrapper(create.frame_group_by_user, username, user_role, "group_by_user")

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
                    create.pie_chart_by_session_state(username, user_role)
                with col7:
                    create.pie_chart_by_job_count(username, user_role)
                with col8:
                    create.efficiency_percentile_chart()

            with tab4:
                st.header("")
                col9, col10 = st.columns(2)
                with col9:
                    date_slider_wrapper(create.bar_char_by_user, username, user_role, "bar_by_user", hyper_threading=True, number_input=True)
                with col10:
                    date_slider_wrapper(create.scatter_chart_data_cpu_gpu_eff, username, user_role, "scatter", hyper_threading=True)

        elif user_role == 'user':
            tab1, tab2 = st.tabs(["Tables", "Charts"]) 
            
            with tab1:
                col1, col2 = st.columns([3, 1])
                with col1: 
                    create.frame_user_all(username, user_role)
                with col2:
                    date_slider_wrapper(create.frame_group_by_user, username, user_role, "by_user")
            with tab2:
                col3, col4 = st.columns(2)
                with col3:
                    create.pie_chart_by_session_state(username, user_role)
                with col4:
                    create.pie_chart_by_job_count(username, user_role)
    else:
        st.title("Login Max-Reports")
        form = st.form(key="login_form")
        
        # Show login form if user is not authenticated
        username = form.text_input("Username")
        password = form.text_input("Password", type="password")
        try:
            if form.form_submit_button("Login"):                
                if authenticate(username, password):
                    st.session_state['username'] = username
                    if is_user_admin(username):
                        st.session_state['user_role'] = 'admin'
                    elif is_user_allowed(username):        
                        st.session_state['user_role'] = 'user'
                    else:
                        st.error("You are not authorized to login")
                        return  # Exit if not authorized
                    
                    st.success('Login successful')
                    st.rerun()  # Re-run to update session state

        except Exception as e:
            st.error("An error occurred during login.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    con = sqlite3.connect('reports.db')
    create = CreateFigures(con)
    st_autorefresh(interval=600000)
    main()



