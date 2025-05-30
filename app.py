import streamlit as st
import pandas as pd
import time
from config import get_config
from datetime import timedelta, datetime
import numpy as np
import sqlite3
import toml
from ldap3 import Server, Connection, ALL
from dataclasses import asdict
from figures import CreateFigures

secrets = toml.load('.streamlit/secrets.toml')

# LDAP Configuration
LDAP_SERVER = secrets['ldap']['server_path']
SEARCH_BASE = secrets['ldap']['search_base']
USE_SSL = secrets['ldap']['use_ssl']

ALLOWED_USERS = secrets['users']['allowed_users']
ADMIN_USERS = secrets['users']['admin_users']
XFEL_USERS = secrets['users']['xfel_users']

def authenticate(username, password):
    if not password:
        st.error("Password cannot be empty")
        return False

    try:
        server = Server(LDAP_SERVER, use_ssl=USE_SSL, get_info=ALL)

        user = f"uid={username},ou=people,ou=rgy,o=desy,c=de"

        conn = Connection(server, user=user, password=password.strip())  
        
        if conn.bind():
            return True  
        else:
            st.error("Invalid username or password") 
            return False  
    except Exception as e:
        st.error(f"LDAP connection error: {e}")
        return False

def is_user_allowed(username):
    return username in ALLOWED_USERS

def is_user_admin(username):
    return username in ADMIN_USERS

def is_user_xfel(username):
    return username in XFEL_USERS

def input_controls():
    
    help_hyper = """some jobs can only use physical cores, therefore hyperthreading cores are not included.  
                    you can click on the checkbox to include hyperthreading cores into the calculations"""
    
    
    
    with st.sidebar:
        
        default_range = [datetime.today() - timedelta(days=30), datetime.today()]
        date_selector = st.date_input("select timerange", default_range, key=f"date_slider")
        
        if len(date_selector) != 2:
            st.stop()

        start_date, end_date = date_selector
        start_date = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_date = int(datetime.combine(end_date, datetime.max.time()).timestamp())


        partition_selector = st.selectbox("select partition", ["All available partitions", "allcpu","allgpu","cfel","cfel-cdi","hzg","maxcpu","cssbgpu","exfel"
                                                                ,"upex","cdcs","cfel-cmi","cfel-ux","com","cssbcpu","exfel-th",
                                                                "exfel-theory","exfel-wp72","jhub","livcpu","livgpu","mcpu","mpa",
                                                                "mpaj","p06","p10","p11","petra4","ponline","ponline_p11",
                                                                "ponline_p11_com","pscpu","psgpu","psxcpu","psxgpu","short",
                                                                "topfgpu","uhhxuv","ukecpu","upex-beamtime","upex-high",
                                                                "upex-middle","xfel-guest","xfel-sim"],
                                                key=f"partition_selector"
                                                )

        if partition_selector == "All available partitions":
                partition_selector = None


        scale = st.checkbox("take hyperthreading cores into account", key=f"checkbox", help=help_hyper)        
        
        if scale:
            scale_efficiency = False

        else:
            scale_efficiency = True




    return start_date, end_date, scale_efficiency, partition_selector

def main():
    if 'user_role' in st.session_state:
        username = st.session_state['username']
        user_role = st.session_state['user_role']
        
        if user_role == 'admin':
            view_options = ["Admin View", "XFEL View", "User View"]
        elif user_role == 'exfel':
            view_options = ["XFEL View", "User View"]
        else:
            view_options = ["User View"]
        
        with st.sidebar:
            selected_view = st.selectbox("Select View", view_options, key="view_select")

        if selected_view == "Admin View":
            user_role = 'admin'
        elif selected_view == "XFEL View":
            user_role = 'exfel'
        else:
            user_role = 'user'

        start_date, end_date, scale_efficiency, partition_selector = input_controls()

        if user_role == 'admin':
            tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data Charts", "Job State Charts", "Overview"])
            with st.spinner("loading..."):
                with tab1:
                    col_num, _ = st.columns([1, 2])
                    col1, col2 = st.columns([3, 1])
                    with col_num:                   
                        number = st.number_input("select last n jobs:", min_value=1, max_value=1_000_000, value=25_000, help=""" Input values above 250k can cause the browser to crash!  
                                                                                                                            Column sorting is disabled for values above 150k!""")
                    with col1:
                        create.frame_user_all(username, user_role, number, partition_selector)
                    with col2:
                        create.frame_group_by_user( start_date, end_date, username, user_role, scale_efficiency, partition_selector)

                with tab2:
                    col_num2, _ = st.columns([1, 2])
                    col3,col4 = st.columns([1,1])
                    
                    with col_num2:
                        number2 = st.number_input("select jobs with a runtime greater than:", min_value=0, value=0)
                    with col3:
                        create.job_counts_by_log2(start_date, end_date, number2, partition_selector)
                    with col4:
                        create.pie_chart_job_runtime(start_date, end_date, scale_efficiency, partition_selector)

                with tab3:
                    col5, col6, col7 = st.columns([1,1,1])
                    with col5:
                        create.pie_chart_by_session_state(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                    with col6: 
                        create.pie_chart_by_job_count(start_date, end_date, username, user_role, partition_selector)
                    with col7:
                        create.pie_chart_batch_inter(start_date, end_date, scale_efficiency, partition_selector)


            with tab4:
                col_num3, _ = st.columns([1, 2])
                col8, col9 = st.columns([1,1])
                with col_num3:
                    number3 = st.number_input("select number of user:", min_value=0, value=20)
                with col8:
                    create.bar_char_by_user(start_date, end_date, username, user_role, number3, scale_efficiency, partition_selector)
                with col9:    
                    create.scatter_chart_data_cpu_gpu_eff(start_date, end_date, username, user_role, scale_efficiency, partition_selector)

        elif user_role == 'exfel':
            tab1, tab2, tab3 = st.tabs(["Tables", "Charts", "Overview"]) 
            with st.spinner("loading"):
                with tab1:
                    col_num, _ = st.columns([1, 2])
                    col1, col2 = st.columns([3, 1])
                    with col_num:                   
                        number = st.number_input("select last n jobs:", min_value=1, max_value=1_000_000, value=25_000, help=""" Input values above 250k can cause the browser to crash!  
                                                                                                                            Column sorting is disabled for values above 150k!""")
                    with col1:
                        create.frame_user_all(username, user_role, number, partition_selector)
                    with col2:
                        create.frame_group_by_user( start_date, end_date, username, user_role, partition_selector)
                with tab2:
                    col3, col4 = st.columns([1,1])
                    with col3:
                        create.pie_chart_by_session_state(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                    with col4: 
                        create.pie_chart_by_job_count(start_date, end_date, username, user_role, partition_selector)
                
                with tab3:
                    col_num2, _ = st.columns([1, 2])
                    col5, col6 = st.columns([1,1])
                    with col_num2:
                        number3 = st.number_input("select number of user:", min_value=0, value=20)
                    with col5:
                        create.bar_char_by_user(start_date, end_date, username, user_role, number3, scale_efficiency, partition_selector)
                    with col6:    
                        create.scatter_chart_data_cpu_gpu_eff(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                
        elif user_role == 'user':    
            tab1, tab2, tab3 = st.tabs(["Tables", "Charts", "Overview"]) 
            with st.spinner("loading"):
                with tab1:
                    col_num, _ = st.columns([1, 2])
                    col1, col2 = st.columns([3, 1])
                    with col_num:                   
                        number = st.number_input("select last n jobs:", min_value=1, max_value=1_000_000, value=25_000, help=""" Input values above 250k can cause the browser to crash!  
                                                                                                                            Column sorting is disabled for values above 150k!""")
                    with col1:
                        create.frame_user_all(username, user_role, number, partition_selector)
                    with col2:
                        create.frame_group_by_user( start_date, end_date, username, user_role, partition_selector)
                with tab2:
                    col3, col4 = st.columns([1,1])
                    with col3:
                        create.pie_chart_by_session_state(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                    with col4: 
                        create.pie_chart_by_job_count(start_date, end_date, username, user_role, partition_selector)
                
                with tab3:
                    col1,col2 = st.columns([1,1])
                    with col1:
                            create.scatter_chart_data_cpu_gpu_eff(start_date, end_date, username, user_role, scale_efficiency, partition_selector)

    else:
        _ , col1, _ = st.columns([1, 2, 1])    
        with col1:    
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
                        elif is_user_xfel(username):
                            st.session_state['user_role'] = 'exfel'
                        elif is_user_allowed(username):        
                            st.session_state['user_role'] = 'user'
                        else:
                            st.error("You are not authorized to login")
                            return  # Exit if not authorized
                        
                        st.rerun()  # Re-run to update session state

            except Exception as e:
                st.error("An error occurred during login.")


if __name__ == "__main__":
    st.set_page_config(layout="wide", 
    page_title="max-reports"
    #initial_sidebar_state="collapsed"
)
    con = sqlite3.connect('max-reports-slurm.sqlite3')
    create = CreateFigures(con)
    main()
