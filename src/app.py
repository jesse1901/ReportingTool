import streamlit as st
from datetime import timedelta, datetime
import sqlite3
import toml

from charts.bar import BarCharts
from charts.pie import PieCharts
from charts.scatter import ScatterCharts
from charts.frames import DataFrames

secrets = toml.load('.streamlit/secrets.toml')

ALLOWED_USERS = secrets['users']['allowed_users']
ADMIN_USERS = secrets['users']['admin_users']
XFEL_USERS = secrets['users']['xfel_users']
UHH_USERS = secrets['users']['uhh_users']

LOGO_URL = secrets['urls']['logo']
ICON_URL = secrets['urls']['icon']


def login():
    if not st.experimental_user.is_logged_in:
        if st.button("Log in with Keycloak"):
            st.login()
        st.stop()



def is_user_allowed(username):
    return username in ALLOWED_USERS

def is_user_admin(username):
    return username in ADMIN_USERS

def is_user_xfel(username):
    return username in XFEL_USERS

def is_user_uhh(username):
    return username in UHH-USERS

def input_controls(user_role=None):
    
    help_hyper = """some jobs can only use physical cores, therefore hyperthreading cores are not included.  
                    you can click on the checkbox to include hyperthreading cores into the calculations"""
    
    
    
    with st.sidebar:
        
        default_range = [datetime.today() - timedelta(days=30), datetime.today()]
        date_selector = st.date_input("select timerange", default_range, key=f"date_slider")
        allowed_groups = None

        if len(date_selector) != 2:
            st.stop()

        start_date, end_date = date_selector
        start_date = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_date = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        if user_role == 'exfel':
             partition_selector = st.selectbox("select partition",  
                                               ["exfel","exfel-th","exfel-theory","exfel-wp72","exrsv", "upex","upex-beamtime","upex-high","upex-middle",
                                                "xfel-guest","xfel-op","xfel-sim" ],   key=f"partition_selector")
        
        elif user_role == 'uhh':
            partition_selector = st.selectbox("select partition", 
            ["acc-uhh","allcpu","allgpu", "maxgpu", "maxcpu"],   key=f"partition_selector")

            allowed_groups = ['i02', 'unihh2']

        else:
            partition_selector = st.selectbox("select partition", 


    ["All available partitions","acc-uhh","allcpu","allgpu","allrsv","cdcs","cfel","cfel-cdi","cfel-cmi","cfel-ux","com",
    "cssbcpu","cssbgpu","exfel","exfel-th","exfel-theory","exfel-wp72","exrsv","fspetra","hzg",
    "jhub","livcpu","livgpu","maxcpu","maxgpu","mcpu","mpa","mpaj","p06","p10","p11","p11x",
    "pcommissioning","petra4","petra4-guest","ponline","ponline_p09","ponline_p11",
    "ponline_p11_com","pscpu","psgpu","psxcpu","psxgpu","short","topfgpu","uhhxuv",
    "ukecpu","upex","upex-beamtime","upex-high","upex-middle","xfel-guest","xfel-op","xfel-sim" ],   key=f"partition_selector")

            
        if partition_selector == "All available partitions":
                partition_selector = None


        scale = st.checkbox("take hyperthreading cores into account", key=f"checkbox", help=help_hyper)        
        
        if scale:
            scale_efficiency = False

        else:
            scale_efficiency = True




    return start_date, end_date, scale_efficiency, partition_selector, allowed_groups

def main():
    if 'user_role' in st.session_state:
        username = st.session_state['username']
        user_role = st.session_state['user_role']
        
        if user_role == 'admin':
            view_options = ["Admin View", "XFEL View", "User View", "UHH View"]
        elif user_role == 'exfel':
            view_options = ["XFEL View", "User View"]
        elif user_role == 'uhh':
            view_options = ["UHH View", "User View"]
        else:
            view_options = ["User View"]
        
        with st.sidebar:
            if st.button("Logout"):
                st.logout()
            selected_view = st.selectbox("select view", view_options, key="view_select")

        if selected_view == "Admin View":
            user_role = 'admin'
        elif selected_view == "XFEL View":
            user_role = 'exfel'
        elif selected_view == "UHH View":
            user_role = 'uhh'
        else:
            user_role = 'user'

        start_date, end_date, scale_efficiency, partition_selector, allowed_groups = input_controls(user_role)

        if user_role == 'admin':
            tab1, tab2, tab3, tab4 = st.tabs(["User Data", "Job Data Charts", "Job State Charts", "Overview"])
            with st.spinner("loading..."):
                with tab1:
                    col_num, col_username, col_jobid, _ = st.columns([1, 1, 1, 2])
                    col1, col2 = st.columns([5, 2])
                    with col_num:                   
                        number = st.number_input("select last n jobs:", min_value=1, max_value=1_000_000, value=25_000, help=""" Input values above 250k can cause the browser to crash!  
                                                                                                                                Column sorting is disabled for values above 150k!""")
                    with col_jobid:
                        filter_jobid = st.text_input("search for JobID", value="", key="jobid_filter", placeholder="<jobID>")
                    with col_username:
                        filter_user = st.text_input("search for User", value="", key="username_filter", placeholder="<username>")
                    with col1:
                        frames.frame_user_all(username, user_role, number, partition_selector, filter_jobid, filter_user, start_date=start_date, end_date=end_date)
                    with col2:
                        frames.frame_group_by_user( start_date, end_date, username, user_role, scale_efficiency, partition_selector)

                with tab2:
                    col_num2, _ = st.columns([1, 2])
                    col3,col4 = st.columns([1,1])
                    
                    with col_num2:
                        number2 = st.number_input("select jobs with a runtime greater than:", min_value=0, value=0)
                    with col3:
                        bar.job_counts_by_log2(start_date, end_date, number2, partition_selector)
                    with col4:
                        pie.pie_chart_job_runtime(start_date, end_date, scale_efficiency, partition_selector)

                with tab3:
                    col5, col6, col7 = st.columns([1,1,1])
                    with col5:
                        pie.pie_chart_by_session_state(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                    with col6: 
                        pie.pie_chart_by_job_count(start_date, end_date, username, user_role, partition_selector)
                    with col7:
                        pie.pie_chart_batch_inter(start_date, end_date, username, user_role, scale_efficiency, partition_selector)


            with tab4:
                col_num3, _ = st.columns([1, 2])
                col8, col9 = st.columns([1,1])
                with col_num3:
                    number3 = st.number_input("select number of user:", min_value=0, value=20)
                with col8:
                    bar.bar_char_by_user(start_date, end_date, username, user_role, number3, scale_efficiency, partition_selector)
                with col9:    
                    scatter.scatter_chart_data_cpu_gpu_eff(start_date, end_date, username, user_role, scale_efficiency, partition_selector)

        elif user_role == 'exfel' or user_role == 'uhh':
            tab1, tab2, tab3, tab4 = st.tabs(["Tables", "Job Data Charts", "Job State Charts", "Overview"]) 
            with st.spinner("loading"):
                with tab1:
                    col_num, col_username, col_jobid,_ = st.columns([1, 1, 1, 2])
                    col1, col2 = st.columns([5, 2])
                    with col_num:                   
                        number = st.number_input("select last n jobs:", min_value=1, max_value=1_000_000, value=25_000, help=""" Input values above 250k can cause the browser to crash!  
                                                                                                                                Column sorting is disabled for values above 150k!""")
                    with col_jobid:
                        filter_jobid = st.text_input("search for JobID", value="", key="jobid_filter", placeholder="<jobID>")
                    with col_username:
                        filter_user = st.text_input("search for User", value="", key="username_filter", placeholder="<username>")
                    with col1:
                        frames.frame_user_all(username, user_role, number, partition_selector, filter_jobid, filter_user, start_date=start_date, end_date=end_date, allowed_groups=allowed_groups)
                    with col2:
                        frames.frame_group_by_user( start_date, end_date, username, user_role, scale_efficiency, partition_selector, allowed_groups)
                
                with tab2:
                    col_num2, _ = st.columns([1, 2])
                    col3,col4 = st.columns([1,1])
                    
                    with col_num2:
                        number2 = st.number_input("select jobs with a runtime greater than:", min_value=0, value=0)
                    with col3:
                        bar.job_counts_by_log2(start_date, end_date, number2, partition_selector, allowed_groups)
                    with col4:
                        pie.pie_chart_job_runtime(start_date, end_date, scale_efficiency, partition_selector, allowed_groups)                
                
                with tab3:
                    col3, col4, col4_5 = st.columns([1,1,1])
                    with col3:
                        pie.pie_chart_by_session_state(start_date, end_date, username, user_role, scale_efficiency, partition_selector, allowed_groups)
                    with col4: 
                        pie.pie_chart_by_job_count(start_date, end_date, username, user_role, partition_selector, allowed_groups)
                    with col4_5:
                        pie.pie_chart_batch_inter(start_date, end_date, username, user_role, scale_efficiency, partition_selector, allowed_groups)
                with tab4:
                    col_num2, _ = st.columns([1, 2])
                    col5, col6 = st.columns([1,1])
                    with col_num2:
                        number3 = st.number_input("select number of user:", min_value=0, value=20)
                    with col5:
                        bar.bar_char_by_user(start_date, end_date, username, user_role, number3, scale_efficiency, partition_selector, allowed_groups)
                    with col6:    
                        scatter.scatter_chart_data_cpu_gpu_eff(start_date, end_date, username, user_role, scale_efficiency, partition_selector, allowed_groups)
                
        elif user_role == 'user':    
            tab1, tab2, tab3 = st.tabs(["Tables", "Charts", "Overview"]) 
            with st.spinner("loading"):
                with tab1:
                    col_num, col_jobid, col_username, _ = st.columns([1, 1, 1, 2])
                    col1, col2 = st.columns([5, 2])
                    with col_num:                   
                        number = st.number_input("select last n jobs:", min_value=1, max_value=1_000_000, value=25_000, help=""" Input values above 250k can cause the browser to crash!  
                                                                                                                                Column sorting is disabled for values above 150k!""")
                    with col_jobid:
                        filter_jobid = st.text_input("search for JobID", value="", key="jobid_filter_user", placeholder="<jobID>")

                    with col1:
                        frames.frame_user_all(username, user_role, number, partition_selector, filter_jobid, filter_user=None, start_date=start_date, end_date=end_date)
                    with col2:
                        frames.frame_group_by_user( start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                with tab2:
                    col3, col4, col4_5 = st.columns([1,1,1])
                    with col3:
                        pie.pie_chart_by_session_state(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                    with col4: 
                        pie.pie_chart_by_job_count(start_date, end_date, username, user_role, partition_selector)
                    with col4_5:
                        pie.pie_chart_batch_inter(start_date, end_date, username, user_role, scale_efficiency, partition_selector)
                with tab3:
                    col1,col2 = st.columns([1,1])
                    with col1:
                            scatter.scatter_chart_data_cpu_gpu_eff(start_date, end_date, username, user_role, scale_efficiency, partition_selector)

    else:
        _ , col1, _ = st.columns([1, 2, 1])    
        with col1:    
            st.title("Login Max-Reports")


            try:
                login()
                username = st.user.preferred_username

                st.session_state['username'] = username

                st.session_state['user_role'] = 'admin'

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
    page_title="max-reports",
    page_icon=ICON_URL
    #initial_sidebar_state="collapsed"
)
    st.logo(
    LOGO_URL,
    icon_image=LOGO_URL,
    size="large"
)

    st.html("""
        <style>

        /* Alternative - alle Logo-Images */
        img[data-testid*="logo"], 
        div[data-testid="stSidebarHeader"] img,
        div[data-testid="stAppViewContainer"] img {
            height: 8rem !important;
            width: auto !important;
            max-width: none !important;
        }

        /* Sidebar Header Container anpassen */
        [data-testid="stSidebarHeader"] {
            height: auto !important;
            padding: 1rem 1rem 2rem 1rem !important;
        }
        </style>
        """)
    con = sqlite3.connect('/var/www/max-reports/ReportingTool/max-reports-slurm2sql-v9.8.sqlite3')
    frames = DataFrames(con)
    bar = BarCharts(con)
    pie = PieCharts(con)
    scatter = ScatterCharts(con)

    main()
