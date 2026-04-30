import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from openai import AzureOpenAI

# 1. KONFIGURASI HALAMAN & META TAG
st.set_page_config(
    page_title="JudolGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Meta tag untuk verifikasi Dicoding
st.markdown("""
    <head>
        <meta name="dicoding:email" content="gevintap@gmail.com">
    </head>
""", unsafe_allow_html=True)

# 2. STYLE KUSTOM (CSS)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
.metric-card {
    background: #1a1d27; border: 1px solid #2d3142;
    border-radius: 12px; padding: 1.2rem 1.5rem;
    text-align: center; margin-bottom: 8px;
}
.metric-label { font-size: 12px; color: #6b7280; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.4rem; }
.metric-value { font-size: 30px; font-weight: 600; line-height: 1.1; }
.metric-sub   { font-size: 12px; color: #6b7280; margin-top:.2rem; }
.badge-low      { background:#064e3b;color:#6ee7b7;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500; }
.badge-medium   { background:#78350f;color:#fcd34d;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500; }
.badge-high     { background:#7c2d12;color:#fb923c;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500; }
.badge-critical { background:#450a0a;color:#f87171;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:500; }
.explanation-box {
    background: #1a1d27; border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0; padding: 1rem 1.2rem;
    font-size: 14px; line-height: 1.7; color: #d1d5db; margin:.8rem 0;
}
.section-title {
    font-size:12px;font-weight:500;color:#6b7280;
    letter-spacing:.1em;text-transform:uppercase;
    margin-bottom:.8rem;margin-top:1.2rem;
}
#MainMenu{visibility:hidden;}footer{visibility:hidden;}header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# 3. KREDENSIAL & INITIALIZATION
# Mengambil secrets dari dashboard hosting (Azure/Streamlit)
try:
    AZURE_KEY = st.secrets["AZURE_KEY"]
    AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
    AZURE_DEPLOY = st.secrets["AZURE_DEPLOY"]
except Exception:
    st.error("Gagal memuat Secrets. Pastikan AZURE_KEY, AZURE_ENDPOINT, dan AZURE_DEPLOY sudah diatur di dashboard hosting.")
    st.stop()

# Inisialisasi client Azure OpenAI
client = AzureOpenAI(api_key=AZURE_KEY, api_version="2024-02-01", azure_endpoint=AZURE_ENDPOINT)

LEVEL_COLORS = {"Low": "#6ee7b7", "Medium": "#fcd34d", "High": "#fb923c", "Critical": "#f87171"}
PROFILE_COLORS = {"normal": "#60a5fa", "early_stage": "#fcd34d", "escalating": "#fb923c", "heavy_gambler": "#f87171"}
PROFILE_FILL = {"normal": "rgba(96,165,250,0.15)", "early_stage": "rgba(252,211,77,0.15)", "escalating": "rgba(251,146,60,0.15)", "heavy_gambler": "rgba(248,113,113,0.15)"}

# 4. LOAD DATA
@st.cache_data
def load_data():
    try:
        # Gunakan relative path. Pastikan folder 'data' ada di root GitHub.
        risk = pd.read_csv("data/risk_scores_with_explanation.csv")
        features = pd.read_csv("data/judolguard_.csv")
        return risk, features
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan di folder 'data/'. Periksa struktur folder GitHub Anda.")
        st.stop()

risk_df, features_df = load_data()

# 5. SIDEBAR & NAVIGATION
with st.sidebar:
    st.markdown("## 🛡️ JudolGuard")
    st.markdown("<p style='font-size:13px;color:#6b7280;margin-top:-8px;'>Early Behavioral Shift Detection</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigasi", ["📊 Overview", "📋 Risk Table", "🔍 Detail Akun"], label_visibility="collapsed")
    st.divider()
    
    sel_levels = st.multiselect("Filter Level", ["Critical","High","Medium","Low"], default=["Critical","High","Medium","Low"])
    available_profiles = sorted(risk_df["profile"].unique()) if "profile" in risk_df.columns else []
    sel_profiles = st.multiselect("Filter Profil", available_profiles, default=available_profiles)
    st.divider()
    st.caption("Microsoft Azure AI Impact Challenge 2025")

filtered = risk_df[risk_df["risk_level"].isin(sel_levels) & risk_df["profile"].isin(sel_profiles)].copy()

# HALAMAN 1 — OVERVIEW (BACK TO ORIGINAL DESIGN)
if page == "📊 Overview":
    st.markdown("# 📊 Overview Dashboard")
    st.markdown("<p style='color:#6b7280;margin-top:-12px;font-size:14px;'>Sistem deteksi dini perubahan perilaku transaksi — tim compliance e-wallet</p>", unsafe_allow_html=True)
    st.divider()

    total      = len(risk_df)
    n_critical = (risk_df["risk_level"] == "Critical").sum()
    n_high     = (risk_df["risk_level"] == "High").sum()
    n_medium   = (risk_df["risk_level"] == "Medium").sum()
    det_rate   = (n_critical + n_high) / total * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics_data = [
        (c1, "Total Akun",   total,             "#a5b4fc", "dianalisis"),
        (c2, "🔴 Critical",  n_critical,        "#f87171", "eskalasi OJK"),
        (c3, "🟠 High",      n_high,            "#fb923c", "batasi transfer"),
        (c4, "🟡 Medium",    n_medium,          "#fcd34d", "notifikasi"),
        (c5, "Detection %",  f"{det_rate:.1f}%", "#34d399", "High+Critical"),
    ]
    for col, label, val, color, sub in metrics_data:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown("<div class='section-title'>Distribusi Risk Level</div>", unsafe_allow_html=True)
        lc    = risk_df["risk_level"].value_counts()
        order = ["Critical", "High", "Medium", "Low"]
        fig   = go.Figure(go.Pie(
            labels=order, values=[lc.get(l, 0) for l in order],
            marker_colors=[LEVEL_COLORS[l] for l in order],
            hole=0.55, textinfo="label+percent"
        ))
        fig.add_annotation(text=f"<b>{total}</b><br>akun", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("<div class='section-title'>Risk Score per Profil</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for p in ["normal","early_stage","escalating","heavy_gambler"]:
            d = risk_df[risk_df["profile"] == p]["final_risk_score"]
            if len(d):
                fig2.add_trace(go.Box(y=d, name=p, marker_color=PROFILE_COLORS[p], line_color=PROFILE_COLORS[p], fillcolor=PROFILE_FILL[p]))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("<div class='section-title'>Temporal Shift Score per Profil</div>", unsafe_allow_html=True)
        if "avg_temporal_shift" in risk_df.columns:
            sm = risk_df.groupby("profile")["avg_temporal_shift"].mean().reset_index()
            sm["color"] = sm["avg_temporal_shift"].apply(lambda x: "#f87171" if x > 0.01 else "#6ee7b7")
            fig3 = px.bar(sm, x="profile", y="avg_temporal_shift", color="color", color_discrete_map="identity", text=sm["avg_temporal_shift"].round(3))
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    with cb:
        st.markdown("<div class='section-title'>Night Ratio per Profil</div>", unsafe_allow_html=True)
        if "avg_night_ratio" in risk_df.columns:
            nm = risk_df.groupby("profile")["avg_night_ratio"].mean().reset_index()
            fig4 = px.bar(nm, x="profile", y="avg_night_ratio", color="avg_night_ratio", color_continuous_scale=["#6ee7b7","#fcd34d","#f87171"], text=nm["avg_night_ratio"].apply(lambda x: f"{x:.2%}"))
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-title'>🚨 Top 10 Akun — Perubahan Pola Transaksi Terbaru</div>", unsafe_allow_html=True)
    if "avg_temporal_shift" in risk_df.columns:
        top10 = risk_df[risk_df["avg_temporal_shift"] > 0].sort_values("avg_temporal_shift", ascending=False).head(10).reset_index(drop=True)
        
        th1, th2, th3, th4, th5, th6 = st.columns([2.2, 1.3, 1.3, 1.5, 1.5, 1.5])
        for col, label in zip([th1,th2,th3,th4,th5,th6], ["Account ID", "Level", "Risk Score", "Shift Score", "Night Ratio", "Burst Score"]):
            with col: st.markdown(f"<div style='font-size:11px;color:#6b7280;font-weight:500;text-transform:uppercase;border-bottom:1px solid #2d3142;'>{label}</div>", unsafe_allow_html=True)

        for rank, (_, row) in enumerate(top10.iterrows(), start=1):
            c1, c2, c3, c4, c5, c6 = st.columns([2.2, 1.3, 1.3, 1.5, 1.5, 1.5])
            with c1: st.markdown(f"<div style='padding:8px 0;'><span style='color:#4b5563;'>#{rank}</span> {row['account_id']}</div>", unsafe_allow_html=True)
            with c2: st.markdown(f"<div style='padding:8px 0;'><span class='badge-{row['risk_level'].lower()}'>{row['risk_level']}</span></div>", unsafe_allow_html=True)
            with c3: st.markdown(f"<div style='padding:8px 0; color:{LEVEL_COLORS[row['risk_level']]};'>{row['final_risk_score']:.1f}/100</div>", unsafe_allow_html=True)
            with c4: st.markdown(f"<div style='padding:8px 0; color:#fb923c;'>+{row['avg_temporal_shift']:.4f}</div>", unsafe_allow_html=True)
            with c5: st.markdown(f"<div style='padding:8px 0;'>{row['avg_night_ratio']:.1%}</div>", unsafe_allow_html=True)
            with c6: st.markdown(f"<div style='padding:8px 0;'>{row['avg_burst_score']:.2f}x</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:0; border-top:0.5px solid #2d3142;'>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='section-title'>Azure AI Stack</div>", unsafe_allow_html=True)
    ca2, cb2, cc2 = st.columns(3)
    stack_data = [
        (ca2, "☁️ Azure OpenAI (GPT-4o)", ["Synthetic data generation","Risk explanation per akun","Dynamic recommendation"]),
        (cb2, "🤖 Azure ML Registry", ["Model: JudolGuard-Behavior v1","MLflow tracking","Workspace: ML_JudolGuard"]),
        (cc2, "🔬 Isolation Forest Pipeline", ["Anomaly detection layer","XGBoost classifier","PR-AUC: 0.9655 | F1: 0.8598"]),
    ]
    for col, title, items in stack_data:
        with col:
            items_html = "".join([f"✓ {i}<br>" for i in items])
            st.markdown(f"""<div class="metric-card" style="text-align:left">
                <div style="color:#a5b4fc;font-weight:500;margin-bottom:6px">{title}</div>
                <div style="font-size:12px;color:#6b7280;line-height:1.8">{items_html}</div>
            </div>""", unsafe_allow_html=True)

# HALAMAN 2 — RISK TABLE (PAGINATION)
elif page == "📋 Risk Table":
    st.markdown("# 📋 Risk Table")
    search = st.text_input("🔍 Cari Account ID", placeholder="Ketik account ID...")
    if search:
        filtered = filtered[filtered["account_id"].str.contains(search, case=False, na=False)]

    st.divider()
    table = filtered.sort_values("final_risk_score", ascending=False).reset_index(drop=True)
    
    # Pagination Logic
    PAGE_SIZE = 20
    total_pages = max(1, int(np.ceil(len(table) / PAGE_SIZE)))
    if "risk_page" not in st.session_state: st.session_state["risk_page"] = 1
    
    curr_page = st.session_state["risk_page"]
    start_idx = (curr_page - 1) * PAGE_SIZE
    shown = table.iloc[start_idx : start_idx + PAGE_SIZE]

    # Header Tabel
    h1, h2, h3 = st.columns([3, 2, 2])
    h1.markdown("**Account ID**")
    h2.markdown("**Score**")
    h3.markdown("**Level**")
    st.divider()

    for _, row in shown.iterrows():
        c1, c2, c3 = st.columns([3, 2, 2])
        c1.write(row["account_id"])
        c2.markdown(f"<span style='color:{LEVEL_COLORS[row['risk_level']]}'>{row['final_risk_score']:.1f}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span class='badge-{row['risk_level'].lower()}'>{row['risk_level']}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:0.5rem 0; border:0.1px solid #2d3142;'>", unsafe_allow_html=True)

    # Navigasi Halaman
    p1, p2, p3 = st.columns([1, 2, 1])
    if p1.button("◀ Prev") and curr_page > 1:
        st.session_state["risk_page"] -= 1
        st.rerun()
    p2.write(f"Halaman {curr_page} dari {total_pages}")
    if p3.button("Next ▶") and curr_page < total_pages:
        st.session_state["risk_page"] += 1
        st.rerun()

# HALAMAN 3 — DETAIL AKUN (ANALISIS AI)
elif page == "🔍 Detail Akun":
    st.markdown("# 🔍 Detail Akun")
    sel_acc = st.selectbox("Pilih Akun", filtered["account_id"].unique())
    acc = risk_df[risk_df["account_id"] == sel_acc].iloc[0]
    
    st.divider()
    
    # KPI Detail
    d1, d2, d3 = st.columns(3)
    d1.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Score</div><div class='metric-value' style='color:{LEVEL_COLORS[acc['risk_level']]}'>{acc['final_risk_score']:.1f}</div></div>", unsafe_allow_html=True)
    d2.markdown(f"<div class='metric-card'><div class='metric-label'>Level</div><div class='metric-value'>{acc['risk_level']}</div></div>", unsafe_allow_html=True)
    d3.markdown(f"<div class='metric-card'><div class='metric-label'>Profil</div><div class='metric-value' style='font-size:22px'>{acc['profile']}</div></div>", unsafe_allow_html=True)

    # Grafik Perilaku (Subplots)
    if features_df is not None:
        acc_f = features_df[features_df["account_id"] == sel_acc].sort_values("day")
        if not acc_f.empty:
            st.markdown("<div class='section-title'>📈 Grafik Pola Perubahan</div>", unsafe_allow_html=True)
            fig_line = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
            fig_line.add_trace(go.Scatter(x=acc_f["day"], y=acc_f["tx_count_24h"], name="Frekuensi", line=dict(color="#fb923c")), row=1, col=1)
            fig_line.add_trace(go.Scatter(x=acc_f["day"], y=acc_f["night_ratio_7d"], name="Night Ratio", line=dict(color="#a78bfa")), row=2, col=1)
            fig_line.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_line, use_container_width=True)

    # AI Analysis Section
    st.markdown("<div class='section-title'>🤖 AI Risk Analysis (Azure OpenAI)</div>", unsafe_allow_html=True)
    if st.button("✨ Generate AI Report", type="primary"):
        with st.spinner("Menganalisis perilaku..."):
            try:
                prompt = f"Analisis akun {sel_acc} dengan risk score {acc['final_risk_score']} dan pola night ratio {acc['avg_night_ratio']:.2%}. Berikan rekomendasi konkret."
                resp = client.chat.completions.create(
                    model=AZURE_DEPLOY,
                    messages=[{"role": "user", "content": prompt}]
                )
                st.markdown(f"<div class='explanation-box'>{resp.choices[0].message.content}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Gagal menghubungi AI: {e}")
