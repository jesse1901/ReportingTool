import streamlit as st

class Documentation:
    @staticmethod
    def documentation():
        st.header("Documentation: Metrics & Fields")
        st.markdown("Below is an explanation of the most relevant columns from the Slurm database.")
        
        st.markdown("---")

        # --- SECTION 1: CPU & TIME ---
        st.subheader("CPU & Time")
        st.markdown("These values help analyze how long a job ran versus how much processing power it actually consumed.")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Time & Reservation**")
            st.markdown("""
            * **Elapsed:** Wall clock time. The actual duration from start to finish.
            
            * **AllocCPUS:** Number of Allocated CPU-Cores

            * **CPU_hours:** Reserved CPU time.
              Formula: Elapsed * Number of CPU-Cores
            
            """)
        
        with c2:
            st.markdown("**Usage & Efficiency**")
            st.markdown("""
            * **CPU_hours_used:** The actual CPU seconds used (sum over all cores).
            
            * **CPU_hours_lost / Lost_CPU_Time:** CPU-Time that is not effectively used. Formula: CPU_hours - CPU_hours_used

            * **CPUEff:** CPU Efficiency. Indicates how well the reserved CPUs were utilized.
              Formula: TotalCPU / CPUTime
            """)

        st.markdown("---")

        # --- SECTION 2: GPU ---
        st.subheader("GPU")
        
        st.markdown("""
        * **ReqGPU / NGpus:** Number of GPUs requested.
        * **GpuUtil:** GPU utilization (normalized).
        * **GpuEff:** Overall GPU efficiency.
          Formula: GpuUtil / (100 * AllocTRES)
        """)
        
        st.markdown("---")
        
        st.subheader("Charts")
        st.markdown("Most of the Charts focus on CPU Utilization, therefore GPU-Utilization is not taken into Account, but could be an argument for bad CPU-Utilization")

        st.markdown("#### Pie Charts")
        st.markdown("""
        * **Lost CPU Time by Job State:** Visualizes the distribution of wasted CPU resources based on the job's final state (e.g., CANCELLED, TIMEOUT, FAILED). This helps identify if specific failure modes are significant contributors to resource loss.
        * **Job Count by Job State:** Displays the total number of jobs categorized by their final state, providing an overview of system usage and job completion rates.
        * **Lost CPU Time by Job Time:** Categorizes lost CPU time based on the duration of the jobs. This helps pinpoint whether short, frequent jobs or long-running jobs are the primary source of inefficiency.
        * **Lost CPU Time by Job Category:** Breaks down wasted resources by job type, distinguishing between Batch jobs and JupyterHub instances.
        """)

        st.markdown("#### Bar Charts")
        st.markdown("""
        * **Total Lost CPU-Time per User:** Ranks users by the total amount of CPU time lost. This is useful for identifying users who might benefit from optimization advice.
        * **Job Count by Job Time:** A histogram showing the frequency of jobs within specific runtime intervals (e.g., jobs running for < 2 minutes vs. > 1 hour).
        """)

        st.markdown("#### Scatter Charts")
        st.markdown("""
        * **CPU & GPU Efficiency by Job Duration:** Plots individual jobs to correlate efficiency with runtime and GPU-Utilization.
            * **X-Axis:** Elapsed Time (Job Duration).
            * **Y-Axis:** CPU Efficiency (%).
            * **Color:** GPU Utilization (%). \n
            This chart visualizes CPU efficiency by job runtime and helps to see the correlation between GPU and CPU efficiency, which can be an explanation for bad CPU utilization.
        """)