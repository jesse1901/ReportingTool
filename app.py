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

def input_wrapper(*functions, current_user, role, key:str, number_input=False, hyper_threading=False, date_select=True,
                        title_number="select number", default_number=20, selector_width=[2,3], col=True, help_date=None, help_num=None, 
                        help_hyper=                                        
                                        '''
                                        some jobs can only use physical cores, therefore only these are taken into account,  
                                        activate the option to also take the hyperthreading cores into account
                                        '''):
    
    col1,col2 = st.columns(selector_width)
    
    with col1:
        if date_select:   
            default_range = [datetime.today() - timedelta(days=30), datetime.today()]
            date_selector = st.date_input("select timerange", default_range, key=f"date_slider_{key}", help=help_date)
            
            if len(date_selector) != 2:
                st.stop()

            start_date, end_date = date_selector
            start_date = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_date = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        if hyper_threading:
            scale_efficiency = True
            scale = st.checkbox("take hyperthreading cores into account", key=f"checkbox_{key}", help=help_hyper)
            
            if scale:
                scale_efficiency = False

            

        if number_input:
            number = st.number_input(title_number, value=default_number, key=f"number_input_{key}", help=help_num )

    if col == False:
        for func in functions:
            if date_select == False and number_input == True:
                func(current_user, role, number)
            elif number_input and hyper_threading:
                func(start_date, end_date, current_user, role, number, scale_efficiency)
            elif number_input:
                func(start_date, end_date, current_user, role, number)
            elif hyper_threading:
                st.markdown("<div style='visibility:hidden;height:75px;'></div>", unsafe_allow_html=True)
                func(start_date, end_date, current_user, role, scale_efficiency)
            else:
                func(start_date, end_date, current_user, role) 

    else: 
        columns = st.columns(len(functions))
        for col, func in zip(columns, functions):
            with col:
                if date_select == False and number_input == True:
                    func(current_user, role, number)
                elif number_input and hyper_threading:
                    func(start_date, end_date, current_user, role, number, scale_efficiency )
                elif number_input:
                    func(start_date, end_date, current_user, role, number)
                elif hyper_threading:
                    st.markdown("<div style='visibility:hidden;height:75px;'></div>", unsafe_allow_html=True)
                    func(start_date, end_date, current_user, role, scale_efficiency)
                else:
                    func(start_date, end_date, current_user, role)

def is_user_allowed(username):
    return username in ALLOWED_USERS

def is_user_admin(username):
    return username in ADMIN_USERS

def is_user_xfel(username):
    return username in XFEL_USERS

def main():
    if 'user_role' in st.session_state:
        username = st.session_state['username']
        user_role = st.session_state['user_role']
        
        if user_role == 'admin':
            tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data Charts", "Job State Charts", "Overview"])
            with st.spinner("loading..."):
                with tab1:
                    #st.header("User Data")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        input_wrapper(create.frame_user_all, current_user=username, role=user_role, 
                                            key= "frame_all", date_select=False, number_input=True, default_number=100000, col=False, title_number="select last n Jobs:",
                                            help_num=
                                                        '''
                                                        input values above 250k can cause the browser to crash!  
                                                        for input values above 150k column sorting is disabled! 
                                                        ''')
                    with col2:
                        input_wrapper(create.frame_group_by_user, current_user=username, role=user_role, key="group_by_user", 
                                            selector_width=[9,1], col=False)

                with tab2:
                    #st.header("Job Data")
                    input_wrapper(
                        create.job_counts_by_log2,
                        create.pie_chart_job_runtime,
                        create.pie_chart_batch_inter,
                        current_user=username, role=user_role, number_input=True, 
                        title_number="select jobs where job time greater than:", key ="tab2_admin", default_number=0, hyper_threading=True)
                with tab3:
                    #st.header("Efficiency")
                    input_wrapper(    
                        create.pie_chart_by_session_state,
                        create.pie_chart_by_job_count,
                        current_user=username, role=user_role, key="tab3_admin", hyper_threading=True)

                with tab4:
                    #st.header("")
                    input_wrapper(create.bar_char_by_user, 
                                        create.scatter_chart_data_cpu_gpu_eff, 
                                        current_user=username, role=user_role,
                                        key="bar_by_user", hyper_threading=True, number_input=True, 
                                        title_number="select number of user:")

        elif user_role == 'exfel':
            tab1, tab2, tab3 = st.tabs(["Tables", "Charts", "Overview"]) 
            with st.spinner("loading"):
                with tab1:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        input_wrapper(create.frame_user_all, current_user=username, role=user_role, 
                                            key= "frame_all_exfel", date_select=False, number_input=True, default_number=100000, col=False, title_number="select last n Jobs:",
                                            help_num=
                                                        '''
                                                        input values above 250k can cause the browser to crash!  
                                                        for input values above 150k column sorting is disabled! 
                                                        ''')
                    with col2:
                        input_wrapper(create.frame_group_by_user, current_user=username, role=user_role, key="group_by_user_exfel", 
                                            selector_width=[9,1], col=False)
                with tab2:
                    input_wrapper(    
                        create.pie_chart_by_session_state,
                        create.pie_chart_by_job_count,
                        current_user=username, role=user_role, key="tab3_exfel", hyper_threading=True)
                
                with tab3:
                    input_wrapper(create.bar_char_by_user, 
                                        create.scatter_chart_data_cpu_gpu_eff, 
                                        current_user=username, role=user_role,
                                        key="bar_by_user_exfel", hyper_threading=True, number_input=True, 
                                        title_number="select number of user:")
                        
        elif user_role == 'user':
            tab1, tab2, tab3 = st.tabs(["Tables", "Charts", "Overview"]) 
            with st.spinner("loading"):
                with tab1:
                    col1, col2 = st.columns([3, 1])
                    with col1: 
                        input_wrapper(create.frame_user_all, current_user=username, role=user_role, 
                                            key= "frame_all_exfel", date_select=False, number_input=True, default_number=100000, col=False, title_number="select last n Jobs:",
                                            help_num=
                                                        '''
                                                        input values above 250k can cause the browser to crash!  
                                                        for input values above 150k column sorting is disabled! 
                                                        ''')
                    with col2:
                        input_wrapper(create.frame_group_by_user, current_user=username, role=user_role, key="group_by_user_admin", 
                                            selector_width=[9,1], col=False)
                with tab2:
                    input_wrapper(    
                        create.pie_chart_by_session_state,
                        create.pie_chart_by_job_count,
                        current_user=username, role=user_role, key="tab3_user", hyper_threading=True)
                with tab3:
                    col1,col2 = st.columns([1,1])
                    with col1:
                        input_wrapper(
                            create.scatter_chart_data_cpu_gpu_eff, 
                            current_user=username, role=user_role,
                            key="bar_by_user_admin", hyper_threading=True,
                            title_number="select number of user:")

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
)
    con = sqlite3.connect('max-reports-slurm.sqlite3')
    create = CreateFigures(con)
    main()


