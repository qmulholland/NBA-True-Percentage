import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
from huggingface_hub import hf_hub_download

# --- PAGE SETUP ---
st.set_page_config(page_title="NBA Pressure Analysis", layout="wide")
st.title("NBA True Clutch & Pressure-Adjusted FT% (High Performance)")

# --- CONFIGURATION ---
REPO_ID = "qpmulho/nba-data-repo" 
DB_FILENAME = "nba.sqlite"

@st.cache_resource
def get_connection():
    db_path = os.path.join(os.getcwd(), DB_FILENAME)
    
    if not os.path.exists(db_path):
        try:
            with st.spinner("Downloading database..."):
                temp_path = hf_hub_download(repo_id=REPO_ID, filename=DB_FILENAME, repo_type="dataset")
                shutil.copy2(temp_path, db_path)
        except Exception as e:
            st.error(f"Failed to download: {e}")
            return None
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # --- PERFORMANCE: CREATE INDEX ---
    # This runs once to ensure searches are instant
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_full_name ON player(full_name);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_player ON play_by_play(player1_id);")
        conn.commit()
    except:
        pass # Index likely already exists
        
    return conn

# --- HIGH PERFORMANCE WIN SIMULATION (Vectorized) ---
def simulate_game_win_fast(margin, seconds_left, trials=10000):
    if seconds_left <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    
    # Average 13 seconds per possession
    possessions = max(1, int(seconds_left / 13))
    
    # Vectorized: Generate all trials and possessions at once in a single matrix
    # This replaces slow Python for-loops with fast C-based NumPy operations
    team_a_pts = np.random.poisson(1.08, (trials, possessions)).sum(axis=1)
    team_b_pts = np.random.poisson(1.08, (trials, possessions)).sum(axis=1)
    
    return np.mean((margin + team_a_pts - team_b_pts) > 0)

# --- CACHED DATA PROCESSING ---
@st.cache_data
def get_processed_player_data(selected_player, _conn):
    query = """
    SELECT period, scoremargin, pctimestring, homedescription, visitordescription
    FROM play_by_play pbp
    JOIN player p ON pbp.player1_id = p.id
    WHERE p.full_name = ? 
    AND (homedescription LIKE '%Free Throw%' OR visitordescription LIKE '%Free Throw%')
    """
    df = pd.read_sql(query, _conn, params=[selected_player])
    
    if df.empty:
        return None

    results = []
    # Process each shot
    for _, row in df.iterrows():
        try:
            # Quick parsing
            m_str = str(row['scoremargin']).upper()
            margin = 0 if any(x in m_str for x in ['TIE', 'NONE', 'NAN']) else int(m_str)
            t_parts = row['pctimestring'].split(':')
            total_sec = int(t_parts[0]) * 60 + int(t_parts[1])
            
            # Outcome
            is_make = 0 if "MISS" in (str(row['homedescription']) + str(row['visitordescription'])).upper() else 1
            
            # Leverage via Fast Sim
            wp_make = simulate_game_win_fast(margin + 1, total_sec)
            wp_miss = simulate_game_win_fast(margin, total_sec)
            leverage = abs(wp_make - wp_miss)
            
            results.append({
                'is_make': is_make, 
                'weight': np.sqrt(leverage) + 0.05, 
                'leverage': leverage,
                'period': row['period']
            })
        except: continue
        
    return pd.DataFrame(results)

# --- APP INTERFACE ---
conn = get_connection()

if conn:
    player_list = pd.read_sql("SELECT DISTINCT full_name FROM player ORDER BY full_name", conn)
    selected_player = st.sidebar.selectbox("Select Player", player_list['full_name'])

    if selected_player:
        # Using the Cached Processing Function
        res_df = get_processed_player_data(selected_player, conn)

        if res_df is not None:
            # Analytics
            raw_avg = res_df['is_make'].mean() * 100
            pressure_avg = (res_df['is_make'] * res_df['weight']).sum() / res_df['weight'].sum() * 100
            clutch_pct = res_df[res_df['leverage'] > 0.15]['is_make'].mean() * 100

            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw FT%", f"{raw_avg:.1f}%")
            c2.metric("Pressure-Adjusted", f"{pressure_avg:.1f}%", f"{pressure_avg - raw_avg:+.1f}%")
            c3.metric("True Clutch FT% (>15% Win Probability Swing)", f"{clutch_pct:.1f}%")

            # Chart
            st.subheader("Accuracy by Period")
            chart_data = res_df.copy()
            chart_data['period_label'] = chart_data['period'].apply(lambda x: str(int(x)) if x < 4 else "4+")
            final_chart = chart_data.groupby('period_label')['is_make'].mean() * 100
            
            st.bar_chart(final_chart.reindex(["1", "2", "3", "4+"]).fillna(0))
        else:
            st.warning("No data found.")