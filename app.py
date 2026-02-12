import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download

# --- PAGE SETUP ---
st.set_page_config(page_title="NBA Stochastic Pressure", layout="wide")
st.title("NBA True Clutch and Stochastic FT Analysis")

# --- DATA DOWNLOAD & CONNECTION ---
# Replace with your Hugging Face username and the dataset repository name
REPO_ID = "qpmulho/nba-data-repo" 
DB_FILENAME = "nba.sqlite"

@st.cache_resource
def get_connection():
    # If the file isn't present, download it from Hugging Face
    if not os.path.exists(DB_FILENAME):
        try:
            with st.spinner("Fetching database from Hugging Face..."):
                path = hf_hub_download(repo_id=REPO_ID, filename=DB_FILENAME, repo_type="dataset")
                # Move the downloaded file into the current working directory
                os.rename(path, DB_FILENAME)
        except Exception as e:
            st.error(f"Failed to download database: {e}")
            return None
            
    return sqlite3.connect(DB_FILENAME, check_same_thread=False)

def simulate_game_win(margin, seconds_left, trials=10000):
    if seconds_left <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    
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

                with st.spinner(f"Analyzing {len(df)} shots for {selected_player}..."):
                    for i, row in df.iterrows():
                        try:
                            m_str = str(row['scoremargin'])
                            margin = 0 if any(x in m_str.upper() for x in ['TIE', 'NONE', 'NAN']) else int(m_str)
                            time_parts = row['pctimestring'].split(':')
                            total_sec = int(time_parts[0]) * 60 + int(time_parts[1])
                            
                            wp_make = simulate_game_win(margin + 1, total_sec)
                            wp_miss = simulate_game_win(margin, total_sec)
                            
                            raw_leverage = abs(wp_make - wp_miss)
                            # Leverage Smoothing
                            weight = np.sqrt(raw_leverage) + 0.05 
                            
                            results.append({
                                'is_make': row['is_make'], 
                                'weight': weight, 
                                'raw_leverage': raw_leverage,
                                'period': row['period']
                            })
                        except: continue

                res_df = pd.DataFrame(results)
                raw_avg = res_df['is_make'].mean() * 100
                stochastic_avg = (res_df['is_make'] * res_df['weight']).sum() / res_df['weight'].sum() * 100
                
                # True Clutch Filter (>15% Win Probability impact)
                clutch_df = res_df[res_df['raw_leverage'] > 0.15]
                true_clutch_pct = clutch_df['is_make'].mean() * 100 if not clutch_df.empty else 0

                # Metrics Display
                col1, col2, col3 = st.columns(3)
                col1.metric("Raw Career FT%", f"{raw_avg:.1f}%")
                col2.metric("Stochastic FT%", f"{stochastic_avg:.1f}%", f"{stochastic_avg - raw_avg:.1f}%")
                col3.metric("True Clutch FT%", f"{true_clutch_pct:.1f}%")

                st.subheader("Accuracy by Quarter")
                quarterly = res_df.groupby('period')['is_make'].mean() * 100
                st.bar_chart(quarterly)
            else:
                st.warning("No shot data found for this player.")
    except Exception as e:
        st.error(f"Error accessing database: {e}")
else:
    st.error("Database connection could not be established.")