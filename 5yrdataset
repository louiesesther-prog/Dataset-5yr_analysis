import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

# --- 1. SETTINGS ---
st.set_page_config(page_title="5-Year Quant Lab", layout="wide")
st.title("🏛️ 5-Year Institutional Spike Analysis")

# --- 2. DATA ENGINE ---
@st.cache_data
def get_unified_data(uploaded_file, ticker_choice):
    try:
        live_df = yf.download(ticker_choice, period='60d', interval='15m', progress=False)
        if isinstance(live_df.columns, pd.MultiIndex):
            live_df.columns = live_df.columns.get_level_values(0)
        live_df = live_df.reset_index()
        live_df.columns.values[0] = 'timestamp'
        live_df.columns = [str(col).lower() for col in live_df.columns]
        live_df['timestamp'] = pd.to_datetime(live_df['timestamp'], utc=True).dt.tz_convert('America/New_York')
    except:
        live_df = pd.DataFrame()

    if uploaded_file is not None:
        try:
            hist_df = pd.read_csv(uploaded_file, sep='\t') 
            hist_df.columns = [str(col).lower().strip() for col in hist_df.columns]
            
            if 'datetime' in hist_df.columns:
                hist_df.rename(columns={'datetime': 'timestamp'}, inplace=True)
            elif 'date' in hist_df.columns and 'time' in hist_df.columns:
                hist_df['timestamp'] = hist_df['date'].astype(str) + ' ' + hist_df['time'].astype(str)
            
            hist_df['timestamp'] = hist_df['timestamp'].astype(str).str.replace('.', '-', regex=False)
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'], errors='coerce', utc=True).dt.tz_convert('America/New_York')
            hist_df = hist_df.dropna(subset=['timestamp'])
            
            if 'tickvolume' in hist_df.columns:
                hist_df['volume'] = hist_df['tickvolume']

            df = pd.concat([hist_df, live_df]).drop_duplicates(subset=['timestamp'])
            st.sidebar.success(f"📈 Total Bars: {len(df):,}")
            return df.sort_values('timestamp')
        except Exception as e:
            st.error(f"CSV Error: {e}")
            return live_df
    return live_df

# --- 3. UI SIDEBAR ---
# Expanded Ticker List
ticker = st.sidebar.selectbox("Select Asset", 
    ["ES=F", "NQ=F", "YM=F", "RTY=F", "GC=F", "CL=F"], 
    index=0,
    help="ES=S&P500, NQ=Nasdaq, YM=Dow, RTY=US2000, GC=Gold, CL=Crude Oil"
)

st.sidebar.info("Upload the 5-year MT5 export for the selected asset below.")
csv_upload = st.sidebar.file_uploader("Upload Historical CSV", type=["csv", "txt"])

z_thresh = st.sidebar.slider("Z-Score Sensitivity", 3.0, 15.0, 5.0)
spike_type = st.sidebar.radio("Directional Filter", ["All", "B-LIQ", "S-LIQ"])

df = get_unified_data(csv_upload, ticker)

# --- 4. ASSET-SPECIFIC QUANT LOGIC ---
if not df.empty:
    # 1. Rolling Stats
    df['vol_mean'] = df['volume'].rolling(window=200).mean()
    df['vol_std'] = df['volume'].rolling(window=200).std()
    df['z_score'] = (df['volume'] - df['vol_mean']) / df['vol_std'].replace(0, np.nan)
    
    # 2. Directional Logic
    df['direction'] = np.where(df['close'] >= df['open'], "Buy-Side", "Sell-Side")
    df['spike_color'] = np.where(df['direction'] == "Buy-Side", "#00FF41", "#FF3131")
    
    # 3. Dynamic Session Labeling
    df['hour_min'] = df['timestamp'].dt.strftime('%H:%M')
    hour = df['timestamp'].dt.hour
    
    if "GC=F" in ticker: # GOLD: High activity usually starts at London Open (3 AM EST)
        is_prime = hour.between(3, 11)
        session_label = "London/NY Overlap"
    elif "CL=F" in ticker: # CRUDE
        is_prime = hour.between(9, 14)
        session_label = "Oil Pit Session"
    else: # INDICES (ES, NQ, YM, RTY)
        is_prime = hour.between(9, 16)
        session_label = "US Core Session"

    df['session'] = np.where(is_prime, session_label, "Off-Peak/Electronic")
    
    # 4. Spike Filtering
    all_spikes = df[df['z_score'] > z_thresh].copy()
    if spike_type == "Buy-Side Only":
        spikes = all_spikes[all_spikes['direction'] == "Buy-Side"].copy()
    elif spike_type == "Sell-Side Only":
        spikes = all_spikes[all_spikes['direction'] == "Sell-Side"].copy()
    else:
        spikes = all_spikes.copy()

    # --- 5. VISUALS ---
    st.subheader(f"Analyzing {ticker} Institutional Order Flow")
    
    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df['timestamp'], y=df['close'], name="Price", line=dict(color='#555', width=1)))
    
    if not spikes.empty:
        fig.add_trace(go.Scattergl(
            x=spikes['timestamp'], y=spikes['close'], mode='markers',
            marker=dict(color=spikes['spike_color'], size=7, symbol='diamond', line=dict(width=1, color='white')),
            name="Spikes",
            customdata=spikes['z_score'],
            hovertemplate="Price: %{y}<br>Z-Score: %{customdata:.2f}<extra></extra>"
        ))

    fig.update_layout(
        template="plotly_dark", height=600,
        xaxis=dict(rangeslider=dict(visible=True), type="date",
        rangeselector=dict(buttons=list([
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])))
    )
    st.plotly_chart(fig, use_container_width=True)

    # Timing Chart
    if not spikes.empty:
        st.subheader("📊 Temporal Distribution of Spikes")
        pattern = spikes.groupby(['hour_min', 'session']).size().reset_index(name='count')
        freq_fig = px.bar(pattern, x='hour_min', y='count', color='session',
                          color_discrete_map={session_label: "#00F2FF", "Off-Peak/Electronic": "#7000FF"})
        freq_fig.update_layout(template="plotly_dark", xaxis={'categoryorder':'total descending'}, height=400)
        st.plotly_chart(freq_fig, use_container_width=True)
else:
    st.warning("Please upload the MT5 CSV file for the selected asset.")
