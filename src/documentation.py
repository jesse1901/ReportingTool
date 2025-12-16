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
        st.subheader("GPU (Accelerators)")
        
        st.markdown("""
        * **ReqGPU / NGpus:** Number of GPUs requested.
        * **GpuUtil:** GPU utilization (normalized).
        * **GpuEff:** Overall GPU efficiency.
          Formula: GpuUtil / (100 * AllocTRES)
        """)
        
        st.markdown("---")
