import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
from huggingface_hub import hf_hub_download

# --- PAGE SETUP ---
st.set_page_config(page_title="NBA Pressure Analysis", layout="wide")
st.title("NBA True Clutch and Pressure-Adjusted FT Analysis")

# --- DATA DOWNLOAD & CONNECTION ---
# Update these to match your Hugging Face Dataset repo
REPO_ID = "qpmulho/nba-data-repo" 
DB_FILENAME = "nba.sqlite"

@st.cache_resource
def get_connection():
    # Use absolute path to ensure SQLite finds the file in the app directory
    db_path = os.path.join(os.getcwd(), DB_FILENAME)
    
    if not os.path.exists(db_path):
        try:
            with st.spinner("Downloading database from Hugging Face..."):
                # Fetch token from Streamlit Secrets
                token = st.secrets["HF_TOKEN"]
                
                # Download file to temporary cache
                temp_path = hf_hub_download(
                    repo_id=REPO_ID, 
                    filename=DB_FILENAME, 
                    repo_type="dataset",
                    token=token
                )
                
                # Copy to local directory (more robust than move for cross-device links)
                shutil.copy2(temp_path, db_path)
                
                if not os.path.exists(db_path):
                    st.error("Database file could not be created in the local directory.")
                    return None
        except Exception as e:
            st.error(f"Failed to download database: {e}")
            return None
            
    # Connect using the absolute path
    return sqlite3.connect(db_path, check_same_thread=False)

def simulate_game_win(margin, seconds_left, trials=10000):
    if seconds_left <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    
    # Approx 13 seconds per possession
    possessions = max(1, int(seconds_left / 13))
    team_a_pts = np.random.poisson(1.08, (trials, possessions)).sum(axis=1)
    team_b_pts = np.random.poisson(1.08, (trials, possessions)).sum(axis=1)
    return np.mean((margin + team_a_pts - team_b_pts) > 0)

# --- APP INTERFACE ---
conn = get_connection()

if conn:
    try:
        player_list = pd.read_sql("SELECT DISTINCT full_name FROM player ORDER BY full_name", conn)
        selected_player = st.sidebar.selectbox("Select a Player", player_list['full_name'])

        if selected_player:
            query = """
            SELECT period, scoremargin, pctimestring, homedescription, visitordescription
            FROM play_by_play pbp
            JOIN player p ON pbp.player1_id = p.id
            WHERE p.full_name = ? 
            AND (homedescription LIKE '%Free Throw%' OR visitordescription LIKE '%Free Throw%')
            """
            df = pd.read_sql(query, conn, params=[selected_player])

            if not df.empty:
                def identify_outcome(row):
                    desc = (str(row['homedescription']) + " " + str(row['visitordescription'])).upper()
                    return 0 if "MISS" in desc else 1

                df['is_make'] = df.apply(identify_outcome, axis=1)
                results = []

                with st.spinner(f"Processing {len(df)} shots..."):
                    for i, row in df.iterrows():
                        try:
                            m_str = str(row['scoremargin'])
                            margin = 0 if any(x in m_str.upper() for x in ['TIE', 'NONE', 'NAN']) else int(m_str)
                            time_parts = row['pctimestring'].split(':')
                            total_sec = int(time_parts[0]) * 60 + int(time_parts[1])
                            
                            wp_make = simulate_game_win(margin + 1, total_sec)
                            wp_miss = simulate_game_win(margin, total_sec)
                            
                            leverage = abs(wp_make - wp_miss)
                            # Weight using sqrt smoothing to give high-pressure shots more significance
                            weight = np.sqrt(leverage) + 0.05 
                            
                            results.append({
                                'is_make': row['is_make'], 
                                'weight': weight, 
                                'leverage': leverage,
                                'period': row['period']
                            })
                        except: continue

                res_df = pd.DataFrame(results)
                raw_avg = res_df['is_make'].mean() * 100
                pressure_avg = (res_df['is_make'] * res_df['weight']).sum() / res_df['weight'].sum() * 100
                
                # True Clutch: High impact moments (>15% Win Prob swing)
                clutch_df = res_df[res_df['leverage'] > 0.15]
                clutch_pct = clutch_df['is_make'].mean() * 100 if not clutch_df.empty else 0

                # Metrics Layout
                c1, c2, c3 = st.columns(3)
                c1.metric("Standard Career FT%", f"{raw_avg:.1f}%")
                c2.metric("Pressure-Adjusted FT%", f"{pressure_avg:.1f}%", f"{pressure_avg - raw_avg:.1f}% Delta")
                c3.metric("True Clutch FT% (>15% Win Probability Swing)", f"{clutch_pct:.1f}%")

                st.subheader("Performance by Period")
                quarterly = res_df.groupby('period')['is_make'].mean() * 100
                st.bar_chart(quarterly)
                
                st.caption(f"Based on {len(res_df)} career free throw attempts.")
            else:
                st.warning("No shot data found for this player.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.error("Could not connect to database. Check your Streamlit Secrets for HF_TOKEN.")