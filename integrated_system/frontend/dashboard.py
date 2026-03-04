import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(layout="wide", page_title="AI Video Analytics Dashboard")

API_URL = "http://localhost:8000"

st.title("🖥️ AI Video Analytics System")
st.markdown("Real-time integrated platform for Vehicle & People Analytics and ANPR.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Feed")
    # Stream the MJPEG video feed
    st.image(f"{API_URL}/video_feed", use_container_width=True)

with col2:
    st.header("Live Analytics")
    
    stats_placeholder = st.empty()
    plates_placeholder = st.empty()
    
    while True:
        try:
            # Fetch stats from FastAPI backend
            res_stats = requests.get(f"{API_URL}/stats")
            res_plates = requests.get(f"{API_URL}/recent_plates")
            res_frs = requests.get(f"{API_URL}/frs_logs")
            
            if res_stats.status_code == 200:
                data = res_stats.json()
                
                with stats_placeholder.container():
                    st.subheader("Vehicle Count")
                    v_cols = st.columns(2)
                    v_cols[0].metric("Outgoing (North)", data["vehicles"]["north"])
                    v_cols[1].metric("Incoming (South)", data["vehicles"]["south"])
                    
                    st.subheader("People Count & Demographics")
                    p_cols = st.columns(4)
                    p_cols[0].metric("Entering", data["people"]["entering"])
                    p_cols[1].metric("Exiting", data["people"]["exiting"])
                    p_cols[2].metric("Male", data["gender"]["male"])
                    p_cols[3].metric("Female", data["gender"]["female"])
                    
            if res_plates.status_code == 200:
                plates = res_plates.json()
                with plates_placeholder.container():
                    st.subheader("Recent License Plates")
                    if plates:
                        df = pd.DataFrame(plates)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.write("No plates detected yet.")
            
            if res_frs.status_code == 200:
                frs_logs = res_frs.json()
                with plates_placeholder.container():  # Actually appending inside the same column structure but conceptually distinct
                    st.subheader("Facial Recognition Logs")
                    if frs_logs:
                        df_frs = pd.DataFrame(frs_logs)
                        st.dataframe(df_frs, use_container_width=True, hide_index=True)
                    else:
                        st.write("No subjects recognized yet.")
                        
        except requests.exceptions.ConnectionError:
            with stats_placeholder.container():
                 st.warning("Cannot connect to backend API. Retrying...")
                 
        time.sleep(2)
