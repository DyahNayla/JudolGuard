import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from openai import AzureOpenAI

# 1. KONFIGURASI HALAMAN & META TAG DICODING
st.set_page_config(
    page_title="JudolGuard",
    page_icon="🛡️",
    layout="wide"
)

# Menambahkan Meta Tag Dicoding agar terbaca oleh bot verifikasi
st.markdown("""
    <head>
        <meta name="dicoding:email" content="gevintap@gmail.com">
    </head>
""", unsafe_allow_html=True)

# 2. LOAD SECRETS (Diambil dari Dashboard Streamlit Cloud/Azure, bukan file)
try:
    AZURE_KEY = st.secrets["AZURE_KEY"]
    AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
    AZURE_DEPLOY = st.secrets["AZURE_DEPLOY"]
except Exception as e:
    st.error("Secrets belum dikonfigurasi di Dashboard Hosting!")
    st.stop()

# 3. INITIALIZE AZURE CLIENT
client = AzureOpenAI(
    api_key=AZURE_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_ENDPOINT
)

# 4. LOAD DATA (Gunakan Path Relatif)
@st.cache_data
def load_data():
    # Pastikan file ini ada di folder 'data' di GitHub kamu
    risk_df = pd.read_csv('data/risk_scores_with_explanation.csv')
    features_df = pd.read_csv('data/judolguard_features.csv')
    return risk_df, features_df

try:
    risk_df, features_df = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}. Pastikan folder 'data' sudah diupload ke GitHub.")
    st.stop()

# 5. UI DASHBOARD (LOGIKA UTAMA)
st.title("🛡️ JudolGuard: Behavioral Shift Detection")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    page = st.selectbox("Pilih Halaman", ["Overview", "Detail Akun"])
    st.info("Microsoft Azure AI Impact Challenge 2025")

if page == "Overview":
    # --- BAGIAN OVERVIEW ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Akun", len(risk_df))
    col2.metric("Critical Risk", len(risk_df[risk_df['risk_level'] == 'Critical']))
    col3.metric("Detection Rate", f"{(len(risk_df[risk_df['risk_score_max'] > 50]) / len(risk_df) * 100):.1f}%")
    
    st.subheader("Distribusi Risiko")
    fig = px.pie(risk_df, names='risk_level', color='risk_level',
                 color_discrete_map={'Critical':'#f87171', 'High':'#fb923c', 'Medium':'#fcd34d', 'Low':'#6ee7b7'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "Detail Akun":
    # --- BAGIAN DETAIL & AI REPORT ---
    selected_acc = st.selectbox("Pilih ID Akun", risk_df['account_id'].unique())
    acc_data = risk_df[risk_df['account_id'] == selected_acc].iloc[0]
    
    st.subheader(f"Analisis Akun: {selected_acc}")
    
    # Tombol AI Report
    if st.button("✨ Generate AI Analysis"):
        with st.spinner("Menganalisis pola dengan Azure OpenAI..."):
            try:
                prompt = f"Analisis risiko judi online untuk akun dengan skor {acc_data['final_risk_score']} dan level {acc_data['risk_level']}."
                response = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}]
                )
                st.info(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Gagal menghubungi Azure: {e}")

# Footer CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stMetricValue"] { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)
