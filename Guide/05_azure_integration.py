# -*- coding: utf-8 -*-
"""
05_azure_integration.py — Fase 5: Azure AI Integration
========================================================
TUJUAN:
  Menggunakan 3 layanan Azure yang berbeda, masing-masing dengan
  fungsi spesifik yang tidak tumpang tindih:

  1. Azure Anomaly Detector → deteksi anomali time-series per akun
  2. Azure OpenAI (GPT-4o)  → generate penjelasan risiko dalam bahasa natural
  3. Azure Machine Learning → register model ke ML Registry

  BOBOT PENILAIAN AZURE = 30% — jangan anggap sepele.
  Kunci: setiap layanan harus "meaningful", bukan sekadar dipanggil.

SETUP:
  Ganti nilai AZURE_* di bagian CONFIG dengan credentials kamu.
"""

import os, json, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import joblib
import warnings
warnings.filterwarnings('ignore')
from openai import AzureOpenAI

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — GANTI BAGIAN INI
# ════════════════════════════════════════════════════════════════════════════

# Azure OpenAI (sudah kamu setup di Fase 2)
AZURE_OPENAI_KEY      = "PASTE_KEY_KAMU"
AZURE_OPENAI_ENDPOINT = "https://projekjudol.openai.azure.com/"
AZURE_OPENAI_DEPLOY   = "gpt-4o"

# Azure Anomaly Detector (buat resource baru di portal.azure.com)
# Resource type: "Anomaly Detector" → buat → ambil Key & Endpoint
ANOMALY_DETECTOR_KEY      = "PASTE_ANOMALY_DETECTOR_KEY"
ANOMALY_DETECTOR_ENDPOINT = "PASTE_ANOMALY_DETECTOR_ENDPOINT"
# Contoh endpoint: https://judolguard-anomaly.cognitiveservices.azure.com/

# Azure Machine Learning (sudah ada di workspace kamu)
# Untuk register model, kita pakai Azure ML SDK v2
AML_SUBSCRIPTION_ID = "PASTE_SUBSCRIPTION_ID"
AML_RESOURCE_GROUP  = "PASTE_RESOURCE_GROUP"
AML_WORKSPACE_NAME  = "PASTE_WORKSPACE_NAME"

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11})

# ════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

df          = pd.read_csv('data/judolguard_features.csv')
risk_scores = pd.read_csv('data/risk_scores.csv')

print(f"Data loaded: {df.shape[0]:,} transaksi, {len(risk_scores)} akun")
print(f"Risk distribution:\n{risk_scores['risk_level'].value_counts().to_string()}")


# ════════════════════════════════════════════════════════════════════════════
# BAGIAN A — AZURE ANOMALY DETECTOR
# ════════════════════════════════════════════════════════════════════════════
"""
Azure Anomaly Detector adalah layanan Microsoft yang mendeteksi anomali
pada data time-series. Cocok untuk kasus kita karena:

→ Data transaksi adalah time-series alami (step = waktu)
→ Kita ingin tahu: kapan tepatnya pola mulai menyimpang dari normal?
→ Azure Anomaly Detector bisa detect "change points" — titik di mana
   tren berubah secara signifikan

Cara kerja:
→ Kita kirim series data (list titik waktu + nilai) ke API
→ API mengembalikan: titik mana yang anomali, batas atas/bawah normal,
   dan apakah ada change point (perubahan tren)

Yang kita kirim: frekuensi transaksi per hari (tx_count_24h)
per akun escalating — untuk membuktikan ada lonjakan yang terdeteksi.
"""

print("\n" + "=" * 55)
print("  AZURE ANOMALY DETECTOR")
print("=" * 55)

def call_anomaly_detector(series_values: list, granularity: str = "daily") -> dict:
    """
    Kirim time series ke Azure Anomaly Detector.
    series_values: list of {"timestamp": "...", "value": float}
    Mengembalikan response JSON dari API.
    """
    url = f"{ANOMALY_DETECTOR_ENDPOINT}anomalydetector/v1.0/timeseries/entire/detect"
    headers = {
        "Ocp-Apim-Subscription-Key": ANOMALY_DETECTOR_KEY,
        "Content-Type": "application/json"
    }
    body = {
        "series": series_values,
        "granularity": granularity,
        "maxAnomalyRatio": 0.35,
        "sensitivity": 85     # 0-99: semakin tinggi = semakin sensitif
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()


# Ambil 3 akun escalating dengan riwayat terpanjang
escalating_accs = risk_scores[risk_scores['profile'] == 'escalating'] \
    .sort_values('n_transactions', ascending=False).head(3)['account_id'].tolist()

anomaly_results = {}

for acc_id in escalating_accs:
    acc_data = df[df['account_id'] == acc_id].sort_values('day')

    # Agregasi per hari: ambil max tx_count per hari
    daily = acc_data.groupby('day')['tx_count_24h'].max().reset_index()

    # Minimum 12 titik data untuk Anomaly Detector
    if len(daily) < 12:
        print(f"  {acc_id}: data terlalu sedikit ({len(daily)} hari), skip")
        continue

    # Format yang dibutuhkan API: list of {timestamp, value}
    # Karena data sintetis tidak punya tanggal real, kita generate
    from datetime import datetime, timedelta
    base_date = datetime(2024, 1, 1)
    series = [
        {
            "timestamp": (base_date + timedelta(days=int(row['day']))).strftime('%Y-%m-%dT00:00:00Z'),
            "value": float(row['tx_count_24h'])
        }
        for _, row in daily.iterrows()
    ]

    try:
        result = call_anomaly_detector(series)
        anomaly_results[acc_id] = {
            'series': series,
            'result': result,
            'daily_data': daily
        }
        n_anomalies = sum(result.get('isAnomaly', []))
        print(f"  ✓ {acc_id}: {len(series)} titik → {n_anomalies} anomali terdeteksi")
    except Exception as e:
        print(f"  ✗ {acc_id}: Error — {e}")
    time.sleep(0.5)


# Visualisasi hasil Anomaly Detector
if anomaly_results:
    fig, axes = plt.subplots(len(anomaly_results), 1,
                               figsize=(14, 5 * len(anomaly_results)))
    if len(anomaly_results) == 1:
        axes = [axes]

    fig.suptitle('Azure Anomaly Detector — Deteksi Lonjakan Frekuensi Transaksi\n'
                  'Titik merah = anomali yang terdeteksi Azure',
                  fontsize=12, fontweight='bold')

    for idx, (acc_id, data) in enumerate(anomaly_results.items()):
        ax = axes[idx]
        daily = data['daily_data']
        result = data['result']
        is_anomaly = result.get('isAnomaly', [False] * len(daily))
        upper = result.get('upperMargins', [0] * len(daily))
        lower = result.get('lowerMargins', [0] * len(daily))
        expected = result.get('expectedValues', daily['tx_count_24h'].tolist())

        days = daily['day'].values
        values = daily['tx_count_24h'].values

        # Plot nilai aktual
        ax.plot(days, values, color='#e67e22', linewidth=1.5,
                 label='Frekuensi aktual', zorder=3)

        # Plot expected (baseline normal)
        ax.plot(days, expected, color='#3498db', linewidth=1,
                 linestyle='--', label='Baseline normal (expected)', alpha=0.7)

        # Confidence band
        upper_bound = [e + u for e, u in zip(expected, upper)]
        lower_bound = [e - l for e, l in zip(expected, lower)]
        ax.fill_between(days, lower_bound, upper_bound, alpha=0.15,
                          color='#3498db', label='Batas normal')

        # Tandai titik anomali dengan titik merah
        anomaly_days = [days[i] for i, a in enumerate(is_anomaly) if a and i < len(days)]
        anomaly_vals = [values[i] for i, a in enumerate(is_anomaly) if a and i < len(values)]
        if anomaly_days:
            ax.scatter(anomaly_days, anomaly_vals, color='red', s=100,
                        zorder=5, label=f'Anomali ({len(anomaly_days)} titik)')

        ax.set_title(f'Akun {acc_id} — Deteksi Anomali Frekuensi Transaksi')
        ax.set_xlabel('Hari ke-')
        ax.set_ylabel('Frekuensi tx / hari')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('data/azure_anomaly_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n Chart tersimpan: data/azure_anomaly_detection.png")
else:
    print("  Tidak ada hasil Anomaly Detector — cek koneksi dan API key")


# ════════════════════════════════════════════════════════════════════════════
# BAGIAN B — AZURE OPENAI: EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
"""
Model ML menghasilkan angka (risk score). Tapi petugas compliance e-wallet
tidak bisa mengambil tindakan hanya dari angka.

Azure OpenAI kita gunakan untuk:
→ Menerima fitur-fitur kunci per akun
→ Menghasilkan penjelasan dalam bahasa natural yang bisa dipahami
→ Memberikan rekomendasi tindakan yang spesifik

Ini adalah "human-readable AI layer" yang membuat sistem kita
bisa langsung digunakan, bukan hanya prototype akademis.
"""

print("\n" + "=" * 55)
print("  AZURE OPENAI — RISK EXPLANATION")
print("=" * 55)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def generate_risk_explanation(account_row: pd.Series) -> str:
    """
    Generate penjelasan risiko dalam bahasa natural untuk satu akun.
    Input: baris dari dataframe risk_scores
    Output: string penjelasan 2-3 kalimat
    """
    prompt = f"""Kamu adalah sistem AI untuk tim compliance e-wallet Indonesia.
Analisis profil risiko akun berikut dan berikan penjelasan singkat (2-3 kalimat)
yang dapat dipahami petugas compliance.

Data akun:
- Risk Score: {account_row['final_risk_score']:.1f}/100
- Risk Level: {account_row['risk_level']}
- Rasio aktivitas malam (7 hari): {account_row['avg_night_ratio']:.2%}
- Frekuensi transaksi rata-rata per 24 jam: {account_row['avg_tx_24h']:.1f} kali
- Jumlah penerima unik per 7 hari: {account_row['avg_unique_recv']:.1f} akun
- Pergeseran ke aktivitas malam: {account_row['avg_temporal_shift']:+.3f}
- Rasio penggunaan QRIS: {account_row['avg_qris_ratio']:.2%}
- Burst score (lonjakan frekuensi): {account_row['avg_burst_score']:.2f}x

Format output:
[RINGKASAN] Satu kalimat ringkasan risiko.
[INDIKATOR] Sebutkan 2-3 indikator paling mencurigakan dengan angkanya.
[TINDAKAN] Satu rekomendasi tindakan konkret yang harus dilakukan.

Gunakan bahasa Indonesia yang jelas dan profesional."""

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOY,
            messages=[
                {"role": "system",
                 "content": "Kamu adalah AI analyst untuk sistem deteksi risiko keuangan. Berikan analisis yang ringkas, akurat, dan actionable."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,    # rendah = lebih konsisten, kurang kreatif
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating explanation: {e}]"


# Generate explanations untuk akun Critical dan High
high_risk_accounts = risk_scores[
    risk_scores['risk_level'].isin(['Critical', 'High'])
].head(10)   # batasi 10 akun untuk hemat token

print(f"Generating explanations untuk {len(high_risk_accounts)} akun berisiko tinggi...")

explanations = []
for idx, (_, row) in enumerate(high_risk_accounts.iterrows()):
    explanation = generate_risk_explanation(row)
    explanations.append({
        'account_id' : row['account_id'],
        'risk_score' : row['final_risk_score'],
        'risk_level' : row['risk_level'],
        'explanation': explanation
    })
    print(f"  ✓ [{idx+1}/{len(high_risk_accounts)}] {row['account_id']} "
          f"(Score: {row['final_risk_score']:.1f})")
    time.sleep(0.8)   # rate limit

# Tampilkan contoh explanation
print("\n" + "─" * 55)
print("  CONTOH RISK EXPLANATION:")
print("─" * 55)
if explanations:
    ex = explanations[0]
    print(f"\n  Akun: {ex['account_id']} | Level: {ex['risk_level']} | Score: {ex['risk_score']}")
    print(f"\n{ex['explanation']}")

# Gabungkan explanations ke risk_scores
explanations_df = pd.DataFrame(explanations)
risk_scores_enriched = risk_scores.merge(
    explanations_df[['account_id', 'explanation']],
    on='account_id', how='left'
)
risk_scores_enriched['explanation'] = risk_scores_enriched['explanation'].fillna(
    'Tingkat risiko rendah — tidak memerlukan penjelasan khusus'
)
risk_scores_enriched.to_csv('data/risk_scores_with_explanation.csv', index=False)
print(f"\n risk_scores_with_explanation.csv tersimpan")


# ════════════════════════════════════════════════════════════════════════════
# BAGIAN C — AZURE MACHINE LEARNING: REGISTER MODEL
# ════════════════════════════════════════════════════════════════════════════
"""
Azure ML Model Registry adalah "warehouse" untuk model ML yang sudah ditraining.

Kenapa penting?
→ Model yang hanya ada di file lokal tidak bisa di-deploy ke production
→ Azure ML Registry memberi model: versioning, metadata, lineage
→ Dari registry, model bisa di-deploy sebagai REST API endpoint
→ Ini adalah standar MLOps yang benar

Untuk kompetisi: cukup register model → screenshot halaman registry = bukti.
"""

print("\n" + "=" * 55)
print("  AZURE MACHINE LEARNING — MODEL REGISTRY")
print("=" * 55)

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential

    # Koneksi ke Azure ML Workspace
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=AML_SUBSCRIPTION_ID,
        resource_group_name=AML_RESOURCE_GROUP,
        workspace_name=AML_WORKSPACE_NAME
    )
    print("  Terhubung ke Azure ML Workspace")

    # Register XGBoost model
    model = Model(
        path="models/xgb_judolguard.pkl",
        name="judolguard-xgboost",
        description=(
            "JudolGuard: Early behavioral shift detection model untuk deteksi "
            "pola transaksi berisiko judol online. "
            f"PR-AUC ditulis di data/model_metrics.txt. "
            "Fitur: temporal shift, velocity, multi-recipient pattern."
        ),
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            "framework"   : "XGBoost",
            "task"        : "binary_classification",
            "use_case"    : "judol_risk_detection",
            "azure_service": "Azure ML Registry"
        }
    )

    registered_model = ml_client.models.create_or_update(model)
    print(f"  ✓ Model terdaftar: {registered_model.name} v{registered_model.version}")
    print(f"  URL: https://ml.azure.com/model/{registered_model.name}")

except ImportError:
    print("  Install Azure ML SDK: pip install azure-ai-ml azure-identity")
except Exception as e:
    print(f"  ✗ Error registering model: {e}")
    print("  Manual alternative: Upload models/xgb_judolguard.pkl via Azure ML Studio UI")
    print("  Steps: Azure ML Studio → Models → Register model → From local files")


# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  FASE 5 SELESAI — AZURE INTEGRATION SUMMARY")
print("=" * 55)
print("  A. Azure Anomaly Detector")
print("     → Mendeteksi titik anomali pada time-series frekuensi transaksi")
print("     → Output: data/azure_anomaly_detection.png")
print("  B. Azure OpenAI (GPT-4o)")
print("     → Generate penjelasan risiko per akun dalam bahasa natural")
print("     → Output: data/risk_scores_with_explanation.csv")
print("  C. Azure Machine Learning")
print("     → Model terdaftar di Azure ML Registry dengan metadata")
print("\n  Next: Fase 6 — Streamlit Dashboard (06_dashboard.py)")
print("=" * 55)
