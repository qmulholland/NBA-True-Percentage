import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
import altair as alt
from huggingface_hub import hf_hub_download

# --- PAGE SETUP ---
st.set_page_config(page_title="NBA True FT% Analysis", layout="wide")

# --- UI CLEANUP: REMOVE 'RUNNING' & TOOLBAR ---
st.markdown("""
    <style>
    /* Hides the 'Running...' status indicator */
    [data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }
    /* Hides the top toolbar (Share, GitHub, etc.) */
    header[data-testid="stHeader"] { display: none !important; }
    /* Removes the red line at the top */
    [data-testid="stDecoration"] { display: none !important; }
    /* Fixes padding for a cleaner look */
    .main .block-container { padding-top: 2rem; }
    
    /* CUSTOM DESCRIPTION BOX STYLING */
    .desc-box {
        background-color: #1e2130;
        border-left: 5px solid #60b4ff;
        padding: 1.5rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    .desc-text {
        color: #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("NBA Standard, Pressure-Adjusted, & True Clutch FT%")

# --- DESCRIPTION BOX ---
st.markdown("""
    <div class="desc-box">
        <p class="desc-text">
            This project analyzes career free throw percentage (from the 1946-47 through the 2022-23 season), 
            adjusting for game-state leverage and win probability swings. 
            <strong>Standard FT%</strong> shows raw accuracy, <strong>Pressure-Adjusted</strong> weighs shots by their impact on the outcome, 
            and <strong>True Clutch</strong> measures shots where the result has >15% win probability impact.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
REPO_ID = "qpmulho/nba-data-repo" 
DB_FILENAME = "nba.sqlite"

@st.cache_resource(show_spinner=False)
def get_connection():
    db_path = os.path.join(os.getcwd(), DB_FILENAME)
    if not os.path.exists(db_path):
        try:
            with st.spinner("Preparing database..."):
                temp_path = hf_hub_download(repo_id=REPO_ID, filename=DB_FILENAME, repo_type="dataset")
                shutil.copy2(temp_path, db_path)
        except Exception as e:
            st.error(f"Failed to download: {e}")
            return None
    
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_player_full_name ON player(full_name);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pbp_player ON play_by_play(player1_id);")
        conn.commit()
    except:
        pass
    return conn

# NEW: Optimized Player List Caching
@st.cache_data(show_spinner=False)
def get_player_list(_conn):
    return pd.read_sql("SELECT DISTINCT full_name FROM player ORDER BY full_name", _conn)

def simulate_game_win_fast(margin, seconds_left, trials=10000):
    if seconds_left <= 0:
        return 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
    possessions = max(1, int(seconds_left / 13))
    team_a_pts = np.random.poisson(1.08, (trials, possessions)).sum(axis=1)
    team_b_pts = np.random.poisson(1.08, (trials, possessions)).sum(axis=1)
    return np.mean((margin + team_a_pts - team_b_pts) > 0)

@st.cache_data(show_spinner=False)
def get_processed_player_data(selected_player, _conn):
    query = """
    SELECT period, scoremargin, pctimestring, homedescription, visitordescription
    FROM play_by_play pbp
    JOIN player p ON pbp.player1_id = p.id
    WHERE p.full_name = ? 
    AND (homedescription LIKE '%Free Throw%' OR visitordescription LIKE '%Free Throw%')
    """
    df = pd.read_sql(query, _conn, params=[selected_player])
    if df.empty: return None

    results = []
    for _, row in df.iterrows():
        try:
            m_str = str(row['scoremargin']).upper()
            margin = 0 if any(x in m_str for x in ['TIE', 'NONE', 'NAN']) else int(m_str)
            t_parts = row['pctimestring'].split(':')
            total_sec = int(t_parts[0]) * 60 + int(t_parts[1])
            is_make = 0 if "MISS" in (str(row['homedescription']) + str(row['visitordescription'])).upper() else 1
            wp_make = simulate_game_win_fast(margin + 1, total_sec)
            wp_miss = simulate_game_win_fast(margin, total_sec)
            leverage = abs(wp_make - wp_miss)
            results.append({'is_make': is_make, 'weight': np.sqrt(leverage) + 0.05, 'leverage': leverage, 'period': row['period']})
        except: continue
    return pd.DataFrame(results)

# --- APP INTERFACE ---
conn = get_connection()

if conn:
    # UPDATED: Using the cached list for speed
    player_data = get_player_list(conn)
    
    selected_player = st.sidebar.selectbox(
        "Select Player", 
        player_data['full_name'], 
        index=None, 
        placeholder="Choose a player..."
    )

    if selected_player:
        with st.spinner(f"Retrieving statistics for {selected_player}..."):
            res_df = get_processed_player_data(selected_player, conn)

        if res_df is not None and not res_df.empty:
            raw_avg = res_df['is_make'].mean() * 100
            pressure_avg = (res_df['is_make'] * res_df['weight']).sum() / res_df['weight'].sum() * 100
            clutch_pct = res_df[res_df['leverage'] > 0.15]['is_make'].mean() * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Raw FT%", f"{raw_avg:.1f}%")
            c2.metric("Pressure-Adjusted FT%", f"{pressure_avg:.1f}%", f"{pressure_avg - raw_avg:+.1f}%")
            c3.metric("True Clutch (>15% Win Probability Swing) FT%", f"{clutch_pct:.1f}%")

            st.subheader("Accuracy by Quarter")
            chart_data = res_df.copy()
            chart_data['Quarter'] = chart_data['period'].apply(lambda x: str(int(x)) if x < 4 else "4+")
            final_chart = chart_data.groupby('Quarter')['is_make'].mean().reset_index()
            final_chart['Make %'] = (final_chart['is_make'] * 100).round(1)
            
            bars = alt.Chart(final_chart).mark_bar(color='#60b4ff').encode(
                x=alt.X('Quarter:N', sort=["1", "2", "3", "4+"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Make %:Q', title="Make %"),
                tooltip=['Quarter', 'Make %']
            ).properties(height=400)

            st.altair_chart(bars, use_container_width=True)
        else:
            st.warning(f"No free throw data found for {selected_player}.")
    else:
        st.info("Select a player from the sidebar to view their Pressure-Adjusted statistics.")