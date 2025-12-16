import streamlit as st

class Documentation:
    @staticmethod
    def documentation():
        st.header("📚 Dokumentation: Metriken & Felder")
        st.caption("Erklärung der wichtigsten Kennzahlen aus der Slurm-Datenbank.")
        
        st.markdown("---")

        # --- SEKTION 1: CPU & ZEIT ---
        st.subheader("⚡ CPU & Laufzeit")
        st.markdown("Diese Werte helfen zu verstehen, wie lange ein Job lief und wie viel Rechenleistung er tatsächlich beansprucht hat.")

        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**⏱️ Zeit & Reservierung**")
            st.markdown("""
            * **Elapsed (Wall Clock):** Die tatsächlich verstrichene Zeit von Start bis Ende.
            
            * **CPUTime (Reserved):** Die reservierte Rechenzeit.  
              `Formel: Elapsed × Anzahl CPUs`
            """)
        
        with c2:
            st.markdown("**📈 Nutzung & Effizienz**")
            st.markdown("""
            * **TotalCPU (Used):** Die tatsächlich genutzten CPU-Sekunden (Summe über alle Kerne).
            
            * **CPUEff (Effizienz):** Wie gut wurden die reservierten CPUs genutzt (0.0 - 1.0).  
              `Formel: TotalCPU / CPUTime`
            """)

        st.markdown("---")

        # --- SEKTION 2: SPEICHER ---
        st.subheader("💾 Arbeitsspeicher (RAM)")
        st.markdown("Unterscheidung zwischen angefordertem Limit und tatsächlichem Verbrauch.")

        m1, m2, m3 = st.columns([1, 1, 1])
        
        with m1:
            st.info("**AllocMem (Limit)**\n\nDer dem Job **zugeteilte** Speicher. Dies ist der relevante Wert für die Abrechnung.")
        
        with m2:
            st.success("**TotalMem (Used)**\n\nDer tatsächlich **genutzte** Speicher (MaxRSS) während der Laufzeit.")
            
        with m3:
            st.warning("**MemEff**\n\nVerhältnis von genutztem zu reserviertem Speicher (`Total / Alloc`).")

        st.caption("Zusätzlich: **ReqMemNode** ist der explizit angeforderte Speicher pro Node.")

        st.markdown("---")

        # --- SEKTION 3: GPU ---
        st.subheader("🚀 Grafikkarten (GPU)")
        
        st.markdown("""
        * **ReqGPU / NGpus:** Anzahl der angeforderten GPUs.
        * **GpuUtil:** Die prozentuale Auslastung der GPU.
        * **GpuMem:** Genutzter VRAM (Grafikspeicher).
        * **GpuEff:** Gesamteffizienz (`GpuUtil / (100 * AllocTRES)`).
        """)
        
        st.caption("Hinweis: GPU-Daten basieren oft auf Durchschnittswerten (Ave) oder Summen (Tot) aus dem TRES-Usage Feld.")

        st.markdown("---")

        # --- SEKTION 4: STRUKTUR ---
        st.subheader("🗂 Job-Struktur")
        
        st.markdown("""
        * **ArrayTaskID:** Identifiziert den spezifischen Sub-Job innerhalb eines Job-Arrays.
        * **JobStep:** Der Schritt innerhalb eines Jobs.
            * `batch`: Das Hauptskript.
            * `extern`: SSH-Logins etc.
            * `0, 1...`: Explizite `srun` Schritte.
        """)
