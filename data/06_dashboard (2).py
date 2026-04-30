
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(
    page_title="JudolGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ── Warna (rgba untuk Plotly) ──────────────────────────────
LEVEL_COLORS = {
    "Low"     : "#6ee7b7",
    "Medium"  : "#fcd34d",
    "High"    : "#fb923c",
    "Critical": "#f87171"
}
PROFILE_COLORS = {
    "normal"       : "#60a5fa",
    "early_stage"  : "#fcd34d",
    "escalating"   : "#fb923c",
    "heavy_gambler": "#f87171"
}
# ── PERBAIKAN: rgba() bukan hex 8 digit ──
PROFILE_FILL = {
    "normal"       : "rgba(96,165,250,0.15)",
    "early_stage"  : "rgba(252,211,77,0.15)",
    "escalating"   : "rgba(251,146,60,0.15)",
    "heavy_gambler": "rgba(248,113,113,0.15)"
}

REC_DETAIL = {
    "Low"     : ("✅ Monitor Pasif",   "Tidak ada tindakan segera.",                                 "#064e3b","#6ee7b7"),
    "Medium"  : ("📢 Kirim Notifikasi","Kirim pesan edukasi risiko judol ke nasabah.",               "#78350f","#fcd34d"),
    "High"    : ("🚫 Batasi Transfer", "Terapkan batas harian Rp500.000 + konfirmasi biometrik.",    "#7c2d12","#fb923c"),
    "Critical": ("🔴 Eskalasi Segera", "Freeze transaksi + laporkan ke tim compliance & OJK.",       "#450a0a","#f87171"),
}

# ════════════════════════════════════════════════════════════
# AZURE OPENAI — DYNAMIC RECOMMENDATION
# ════════════════════════════════════════════════════════════
AZURE_KEY      = st.secrets.get("AZURE_KEY", "")
AZURE_ENDPOINT = st.secrets.get("AZURE_ENDPOINT", "")
AZURE_DEPLOY   = st.secrets.get("AZURE_DEPLOY", "gpt-4o")

def get_azure_recommendation(acc_row: pd.Series) -> str:
    """
    Generate rekomendasi tindakan real-time via Azure OpenAI.
    Dipanggil saat user klik tombol di Detail Akun.
    """
    if not AZURE_KEY or not AZURE_ENDPOINT:
        return None  # fallback ke rekomendasi statis

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=AZURE_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_ENDPOINT
        )

        prompt = f"""Kamu adalah sistem AI compliance untuk e-wallet Indonesia yang mendeteksi judi online.
Berikan analisis SINGKAT dan ACTIONABLE (3-4 kalimat) untuk akun berikut:

Data perilaku:
- Risk Score   : {acc_row['final_risk_score']:.1f}/100
- Risk Level   : {acc_row['risk_level']}
- Night ratio  : {acc_row['avg_night_ratio']:.1%} (transaksi tengah malam)
- Frekuensi    : {acc_row['avg_tx_24h']:.1f} transaksi/24 jam
- Penerima unik: {acc_row['avg_unique_recv']:.1f} akun/7 hari
- Temporal shift: {acc_row['avg_temporal_shift']:+.3f} (+ = bergeser ke malam)
- QRIS ratio   : {acc_row['avg_qris_ratio']:.1%}
- Burst score  : {acc_row['avg_burst_score']:.2f}x

Format jawaban:
[ANALISIS] Jelaskan pola perilaku yang terdeteksi.
[INDIKATOR UTAMA] Sebutkan 2 indikator paling mencurigakan dengan angkanya.
[TINDAKAN] Rekomendasikan 1 tindakan konkret yang harus segera dilakukan tim compliance.

Gunakan bahasa Indonesia yang profesional dan ringkas."""

        resp = client.chat.completions.create(
            model=AZURE_DEPLOY,
            messages=[
                {"role": "system", "content": "Kamu analis risiko keuangan digital. Berikan analisis singkat, akurat, dan actionable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"Error koneksi Azure OpenAI: {e}"


# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        risk = pd.read_csv("data/risk_scores_with_explanation.csv")
    except FileNotFoundError:
        try:
            risk = pd.read_csv("data/risk_scores.csv")
            risk["explanation"] = "Klik Generate AI Report untuk analisis."
        except FileNotFoundError:
            st.error("File data/risk_scores.csv tidak ditemukan.")
            st.stop()
    try:
        features = pd.read_csv("data/judolguard_features.csv")
    except FileNotFoundError:
        features = None
    return risk, features

risk_df, features_df = load_data()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ JudolGuard")
    st.markdown("<p style='font-size:13px;color:#6b7280;margin-top:-8px;'>Early Behavioral Shift Detection<br>untuk E-Wallet Indonesia</p>", unsafe_allow_html=True)
    st.divider()

    page = st.radio("Nav", ["📊 Overview","📋 Risk Table","🔍 Detail Akun"], label_visibility="collapsed")
    st.divider()

    st.markdown("**Filter**")
    sel_levels   = st.multiselect("Level",  ["Critical","High","Medium","Low"],   default=["Critical","High","Medium","Low"])
    sel_profiles = st.multiselect("Profil", sorted(risk_df["profile"].unique()),  default=sorted(risk_df["profile"].unique()))
    st.divider()

    # Model metrics
    st.markdown("**Model Performance**")
    try:
        for line in open("data/model_metrics.txt").read().strip().split("\n"):
            if ":" in line:
                k,v = line.split(":",1)
                st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;'><span style='color:#6b7280'>{k.strip()}</span><span style='color:#a5b4fc;font-weight:500'>{v.strip()}</span></div>", unsafe_allow_html=True)
    except: st.caption("model_metrics.txt tidak ditemukan")

    st.divider()
    # Azure status
    if AZURE_KEY:
        st.markdown("<div style='font-size:12px;color:#6ee7b7'>☁️ Azure OpenAI: Terhubung</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:12px;color:#f87171'>☁️ Azure OpenAI: Set di Secrets</div>", unsafe_allow_html=True)
        st.caption("Tambahkan AZURE_KEY di Streamlit Secrets")

    st.divider()
    st.caption("Microsoft Azure AI Impact Challenge 2025")

# Filter
filtered = risk_df[
    risk_df["risk_level"].isin(sel_levels) &
    risk_df["profile"].isin(sel_profiles)
].copy()


# ════════════════════════════════════════════════════════════
# HALAMAN 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Overview Dashboard")
    st.markdown("<p style='color:#6b7280;margin-top:-12px;font-size:14px;'>Sistem deteksi dini perubahan perilaku transaksi — tim compliance e-wallet</p>", unsafe_allow_html=True)
    st.divider()

    total      = len(risk_df)
    n_critical = (risk_df["risk_level"]=="Critical").sum()
    n_high     = (risk_df["risk_level"]=="High").sum()
    n_medium   = (risk_df["risk_level"]=="Medium").sum()
    det_rate   = (n_critical+n_high)/total*100

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,label,val,color,sub in [
        (c1,"Total Akun",  total,       "#a5b4fc","dianalisis"),
        (c2,"🔴 Critical", n_critical,  "#f87171","eskalasi OJK"),
        (c3,"🟠 High",     n_high,      "#fb923c","batasi transfer"),
        (c4,"🟡 Medium",   n_medium,    "#fcd34d","notifikasi"),
        (c5,"Detection %", f"{det_rate:.1f}%","#34d399","High+Critical"),
    ]:
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
        lc = risk_df["risk_level"].value_counts()
        order = ["Critical","High","Medium","Low"]
        fig = go.Figure(go.Pie(
            labels=order, values=[lc.get(l,0) for l in order],
            marker_colors=[LEVEL_COLORS[l] for l in order],
            hole=0.55, textinfo="label+percent",
            textfont=dict(size=12,color="white")
        ))
        fig.add_annotation(text=f"<b>{total}</b><br>akun", x=0.5,y=0.5,font_size=14,font_color="white",showarrow=False)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False,height=280,margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("<div class='section-title'>Risk Score per Profil</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for p in ["normal","early_stage","escalating","heavy_gambler"]:
            d = risk_df[risk_df["profile"]==p]["final_risk_score"]
            if len(d):
                # ── PERBAIKAN: pakai rgba() bukan hex+33 ──
                fig2.add_trace(go.Box(
                    y=d, name=p,
                    marker_color=PROFILE_COLORS[p],
                    line_color=PROFILE_COLORS[p],
                    fillcolor=PROFILE_FILL[p]   # rgba() string
                ))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=280,
            margin=dict(t=10,b=10,l=10,r=10),font=dict(color="#9ca3af",size=11),
            xaxis=dict(gridcolor="#2d3142"),yaxis=dict(gridcolor="#2d3142",title="Risk Score"),showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("<div class='section-title'>Temporal Shift Score per Profil</div>", unsafe_allow_html=True)
        if "avg_temporal_shift" in risk_df.columns:
            sm = risk_df.groupby("profile")["avg_temporal_shift"].mean().reset_index()
            sm.columns=["profile","shift"]
            sm["color"]=sm["shift"].apply(lambda x:"#f87171" if x>0.01 else "#6ee7b7")
            fig3=px.bar(sm,x="profile",y="shift",color="color",color_discrete_map="identity",text=sm["shift"].round(3))
            fig3.add_hline(y=0,line_color="#6b7280",line_dash="dash")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260,
                margin=dict(t=10,b=10,l=10,r=10),font=dict(color="#9ca3af",size=11),showlegend=False,
                xaxis=dict(gridcolor="#2d3142"),yaxis=dict(gridcolor="#2d3142",title="Shift Score"))
            fig3.update_traces(textposition="outside",textfont_color="white")
            st.plotly_chart(fig3, use_container_width=True)

    with cb:
        st.markdown("<div class='section-title'>Night Ratio per Profil</div>", unsafe_allow_html=True)
        if "avg_night_ratio" in risk_df.columns:
            nm=risk_df.groupby("profile")["avg_night_ratio"].mean().reset_index()
            nm.columns=["profile","night_ratio"]
            fig4=px.bar(nm,x="profile",y="night_ratio",color="night_ratio",
                color_continuous_scale=["#6ee7b7","#fcd34d","#f87171"],
                text=nm["night_ratio"].apply(lambda x:f"{x:.2%}"))
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=260,
                margin=dict(t=10,b=10,l=10,r=10),font=dict(color="#9ca3af",size=11),showlegend=False,
                coloraxis_showscale=False,xaxis=dict(gridcolor="#2d3142"),yaxis=dict(gridcolor="#2d3142"))
            fig4.update_traces(textposition="outside",textfont_color="white")
            st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-title'>Azure AI Stack</div>", unsafe_allow_html=True)
    ca2,cb2,cc2=st.columns(3)
    for col,title,items in [
        (ca2,"☁️ Azure OpenAI (GPT-4o)",   ["Synthetic data generation","Risk explanation per akun","Dynamic recommendation (real-time)"]),
        (cb2,"🤖 Azure ML Registry",        ["Model: JudolGuard-Behavior-Model v1","Experiment tracking (MLflow)","Workspace: ML_JudolGuard"]),
        (cc2,"🔬 Isolation Forest Pipeline",["Anomaly detection layer","XGBoost risk classifier","PR-AUC: 0.9655 | F1: 0.8598"]),
    ]:
        with col:
            items_html="".join([f"✓ {i}<br>" for i in items])
            st.markdown(f"""<div class="metric-card" style="text-align:left">
                <div style="color:#a5b4fc;font-weight:500;margin-bottom:6px">{title}</div>
                <div style="font-size:12px;color:#6b7280;line-height:1.8">{items_html}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HALAMAN 2 — RISK TABLE
# ════════════════════════════════════════════════════════════
elif page == "📋 Risk Table":
    st.markdown("# 📋 Risk Table")
    st.markdown(f"<p style='color:#6b7280;margin-top:-12px;font-size:14px;'>Menampilkan {len(filtered):,} akun</p>", unsafe_allow_html=True)

    search=st.text_input("🔍 Cari Account ID", placeholder="Ketik account ID...")
    if search:
        filtered=filtered[filtered["account_id"].str.contains(search,case=False,na=False)]

    st.divider()
    table=filtered.sort_values("final_risk_score",ascending=False).reset_index(drop=True)

    h1,h2,h3,h4,h5=st.columns([2,1.5,1.5,2,2])
    for col,label in zip([h1,h2,h3,h4,h5],["Account ID","Score","Level","Top Trigger","Rekomendasi"]):
        with col:
            st.markdown(f"<div style='font-size:11px;color:#6b7280;font-weight:500;letter-spacing:.08em;text-transform:uppercase;padding:4px 0'>{label}</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:0 0 4px;border:none;border-top:1px solid #2d3142'>", unsafe_allow_html=True)

    for _,row in table.head(50).iterrows():
        level=row["risk_level"]; score=row["final_risk_score"]
        color=LEVEL_COLORS.get(level,"#6b7280")
        c1,c2,c3,c4,c5=st.columns([2,1.5,1.5,2,2])
        with c1: st.markdown(f"<div style='font-weight:500;font-size:13px;color:#e2e8f0;padding:7px 0'>{row['account_id']}</div>",unsafe_allow_html=True)
        with c2: st.markdown(f"<div style='padding:7px 0'><div style='font-size:13px;font-weight:500;color:{color}'>{score:.1f}/100</div><div style='background:#2d3142;border-radius:4px;height:3px;margin-top:4px'><div style='background:{color};width:{score}%;height:3px;border-radius:4px'></div></div></div>",unsafe_allow_html=True)
        with c3: st.markdown(f"<div style='padding:7px 0'><span class='badge-{level.lower()}'>{level}</span></div>",unsafe_allow_html=True)
        with c4: st.markdown(f"<div style='font-size:12px;color:#9ca3af;padding:7px 0'>{row.get('top_triggers','-')}</div>",unsafe_allow_html=True)
        with c5:
            rec=str(row.get("recommendation","-"))
            st.markdown(f"<div style='font-size:12px;color:#9ca3af;padding:7px 0'>{rec[:45]}{'...' if len(rec)>45 else ''}</div>",unsafe_allow_html=True)
        st.markdown("<hr style='margin:0;border:none;border-top:0.5px solid #2d3142'>",unsafe_allow_html=True)

    if len(table)>50: st.caption(f"Menampilkan 50 dari {len(table)} akun.")


# ════════════════════════════════════════════════════════════
# HALAMAN 3 — DETAIL AKUN
# ════════════════════════════════════════════════════════════
elif page == "🔍 Detail Akun":
    st.markdown("# 🔍 Detail Akun")

    opts=filtered.sort_values("final_risk_score",ascending=False)["account_id"].tolist()
    if not opts: st.warning("Tidak ada akun."); st.stop()

    sel=st.selectbox("Pilih Akun", opts,
        format_func=lambda x: f"{x}  —  {risk_df[risk_df['account_id']==x]['risk_level'].values[0]}  ({risk_df[risk_df['account_id']==x]['final_risk_score'].values[0]:.1f})")

    acc=risk_df[risk_df["account_id"]==sel].iloc[0]
    level=acc["risk_level"]; score=acc["final_risk_score"]; color=LEVEL_COLORS.get(level,"#6b7280")

    st.divider()

    h1,h2,h3=st.columns([3,1,1])
    with h1: st.markdown(f"<div style='font-size:22px;font-weight:600;color:#e2e8f0'>{sel}</div><div style='font-size:13px;color:#6b7280;margin-top:4px'>Profil: {acc['profile']} · {acc.get('n_transactions','-')} transaksi</div>",unsafe_allow_html=True)
    with h2: st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Score</div><div class='metric-value' style='color:{color}'>{score:.1f}</div><div class='metric-sub'>dari 100</div></div>",unsafe_allow_html=True)
    with h3: st.markdown(f"<div class='metric-card'><div class='metric-label'>Risk Level</div><div class='metric-value' style='color:{color};font-size:22px'>{level}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # GRAFIK PERUBAHAN PERILAKU
    st.markdown("<div class='section-title'>📅 Timeline Perilaku Transaksi (7 Hari)</div>", unsafe_allow_html=True)
    
    acc_timeline = timeline_df[timeline_df["account_id"] == sel_acc]
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("Frekuensi Tx / 24 Jam", "Night Ratio (Pola Jam Malam)", "Temporal Shift (Pergeseran Perilaku)"),
                        vertical_spacing=0.1)
    
    # Chart 1: Freq
    fig.add_trace(go.Scatter(x=acc_timeline["day"], y=acc_timeline["tx_count_24h"], name="Frekuensi", line=dict(color="#fb923c", width=3), fill='tozeroy'), row=1, col=1)
    # Chart 2: Night Ratio
    fig.add_trace(go.Scatter(x=acc_timeline["day"], y=acc_timeline["night_ratio_7d"], name="Night Ratio", line=dict(color="#818cf8", width=3), fill='tozeroy'), row=2, col=1)
    # Chart 3: Temporal Shift
    fig.add_trace(go.Bar(x=acc_timeline["day"], y=acc_timeline["temporal_shift"], name="Shift", marker_color="#f87171"), row=3, col=1)
    
    fig.update_layout(height=600, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8")
    fig.update_xaxes(gridcolor="#2d3142")
    fig.update_yaxes(gridcolor="#2d3142")
    st.plotly_chart(fig, use_container_width=True)

    # ── TOMBOL GENERATE AI REPORT (Azure OpenAI real-time) ──
    st.markdown("<div class='section-title'>🤖 AI Risk Analysis (Azure OpenAI GPT-4o)</div>", unsafe_allow_html=True)

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        generate_btn = st.button("✨ Generate AI Report", type="primary", use_container_width=True)

    # Session state untuk simpan hasil per akun
    key = f"ai_report_{sel}"
    if key not in st.session_state:
        st.session_state[key] = None

    if generate_btn:
        with st.spinner("Azure OpenAI sedang menganalisis pola perilaku..."):
            result = get_azure_recommendation(acc)
            if result:
                st.session_state[key] = result
            else:
                # Fallback ke explanation yang sudah ada
                st.session_state[key] = acc.get("explanation", "Penjelasan tidak tersedia.")

    # Tampilkan hasil
    if st.session_state[key]:
        st.markdown(f"<div class='explanation-box'>{st.session_state[key]}</div>", unsafe_allow_html=True)
    else:
        exp = acc.get("explanation", "")
        if exp and exp not in ["Klik Generate AI Report untuk analisis.", "Penjelasan belum di-generate.", ""]:
            st.markdown(f"<div class='explanation-box'>{exp}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#1a1d27;border:1px dashed #2d3142;border-radius:8px;padding:1.5rem;text-align:center;color:#6b7280;font-size:13px;'>Klik tombol di atas untuk generate analisis dari Azure OpenAI GPT-4o</div>", unsafe_allow_html=True)

    # ── Behavioral features ──
    st.markdown("<div class='section-title'>📈 Behavioral Features</div>", unsafe_allow_html=True)
    feat_map={
        "avg_night_ratio"   :("Night Ratio 7d",  "%",  100),
        "avg_tx_24h"        :("Frekuensi Tx/24h","x",  1),
        "avg_unique_recv"   :("Penerima Unik/7d","",   1),
        "avg_burst_score"   :("Burst Score",     "x",  1),
        "avg_temporal_shift":("Temporal Shift",  "",   1),
        "avg_qris_ratio"    :("QRIS Ratio",      "%",  100),
    }
    cols=st.columns(3)
    for i,(k,(label,unit,mult)) in enumerate(feat_map.items()):
        if k in acc:
            with cols[i%3]: st.metric(label, f"{acc[k]*mult:.1f}{unit}")

    # ── Timeline ──
    if features_df is not None:
        acc_f=features_df[features_df["account_id"]==sel].sort_values("day")
        if len(acc_f)>0:
            st.markdown("<div class='section-title'>📅 Timeline Perilaku Transaksi</div>", unsafe_allow_html=True)
            fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
                subplot_titles=("Frekuensi Tx/24 Jam","Night Ratio (7 hari)","Temporal Shift Score"),
                vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=acc_f["day"],y=acc_f["tx_count_24h"],fill="tozeroy",
                fillcolor="rgba(251,146,60,0.15)",line=dict(color="#fb923c",width=2)),row=1,col=1)
            fig.add_trace(go.Scatter(x=acc_f["day"],y=acc_f["night_ratio_7d"],fill="tozeroy",
                fillcolor="rgba(167,139,250,0.15)",line=dict(color="#a78bfa",width=2)),row=2,col=1)
            fig.add_hline(y=0.3,line_color="#6b7280",line_dash="dash",row=2,col=1)
            shifts=acc_f["temporal_shift"].values
            fig.add_trace(go.Bar(x=acc_f["day"],y=shifts,
                marker_color=["#f87171" if v>0 else "#6ee7b7" for v in shifts]),row=3,col=1)
            fig.add_hline(y=0,line_color="#6b7280",row=3,col=1)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",height=460,
                showlegend=False,font=dict(color="#9ca3af",size=11),margin=dict(t=30,b=10,l=10,r=10))
            for i in range(1,4):
                fig.update_xaxes(gridcolor="#2d3142",row=i,col=1)
                fig.update_yaxes(gridcolor="#2d3142",row=i,col=1)
            st.plotly_chart(fig,use_container_width=True)

    # ── Rekomendasi tindakan ──
    st.divider()
    st.markdown("<div class='section-title'>⚡ Rekomendasi Tindakan</div>", unsafe_allow_html=True)
    title,desc,bg,fg=REC_DETAIL.get(level,REC_DETAIL["Low"])
    st.markdown(f"<div style='background:{bg};border:1px solid {fg}33;border-radius:10px;padding:1.2rem 1.5rem;'><div style='font-size:16px;font-weight:600;color:{fg};margin-bottom:8px'>{title}</div><div style='font-size:14px;color:#d1d5db;line-height:1.6'>{desc}</div></div>",unsafe_allow_html=True)
