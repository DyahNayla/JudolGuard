# -*- coding: utf-8 -*-
"""
04_modeling.py — Fase 4: ML Modeling
======================================
TUJUAN:
  Membangun dua model yang saling melengkapi:
  1. Isolation Forest  → deteksi anomali (tidak butuh label)
  2. XGBoost Classifier → klasifikasi risiko (butuh label is_at_risk)

  Keduanya digabungkan: score Isolation Forest menjadi fitur tambahan
  untuk XGBoost. Ini meningkatkan akurasi karena model supervised
  diperkuat oleh sinyal unsupervised.

OUTPUT:
  - models/xgb_judolguard.pkl
  - data/risk_scores.csv (risk score per akun untuk dashboard)
  - data/model_metrics.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    f1_score, roc_auc_score
)
import xgboost as xgb

os.makedirs('models', exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 11})


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA & DEFINISI FITUR
# ════════════════════════════════════════════════════════════════════════════
"""
Kita load data hasil feature engineering dari Fase 2.
Fitur yang digunakan dipilih berdasarkan hasil korelasi di EDA (Fase 3):
hanya fitur yang punya relevansi perilaku judol yang masuk.
"""

df = pd.read_csv('data/judolguard_features.csv')
print(f"Data loaded: {df.shape[0]:,} baris, {df['account_id'].nunique()} akun")

# ── Definisi feature columns ────────────────────────────────────────────────
# Penjelasan kenapa kolom ini dipilih:
# - Temporal: jam aktif, rasio malam, pergeseran ke malam
# - Velocity: frekuensi transaksi, lonjakan, total amount
# - Multi-recipient: jumlah penerima unik
# - Channel: rasio QRIS (tanda sembunyikan jejak)
# - Behavioral flags: drain cycle, dormant

FEATURE_COLS = [
    # Temporal
    'hour_of_day', 'is_night', 'night_ratio_7d', 'night_ratio_14d', 'temporal_shift',
    # Velocity
    'amount_log', 'amount_vs_avg_7d', 'total_amount_7d',
    'tx_count_24h', 'tx_count_7d', 'burst_score',
    # Multi-recipient
    'unique_recv_7d', 'unique_recv_24h',
    # Channel
    'qris_ratio_7d',
    # Flags
    'drain_cycle_flag', 'dormant_flag'
]

TARGET_COL = 'is_at_risk'

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET_COL]

print(f"\nFeature matrix: {X.shape}")
print(f"Target distribusi:\n{y.value_counts().to_string()}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ISOLATION FOREST (UNSUPERVISED ANOMALY DETECTION)
# ════════════════════════════════════════════════════════════════════════════
"""
Isolation Forest bekerja dengan cara memisahkan titik data secara acak.
Titik yang mudah diisolasi = anomali (butuh sedikit pemisahan).
Titik yang susah diisolasi = normal (butuh banyak pemisahan).

Kenapa dipakai di sini?
→ Isolation Forest TIDAK butuh label. Dia belajar dari pola data itu sendiri.
→ Score anomali yang dihasilkan menjadi fitur tambahan untuk XGBoost.
→ Ini adalah "early warning layer" sebelum supervised model bekerja.
"""

print("\n" + "=" * 55)
print("  MODEL 1: ISOLATION FOREST")
print("=" * 55)

# Train HANYA pada data normal (is_at_risk == 0)
# Karena kita ingin model belajar apa itu "normal",
# lalu mendeteksi yang menyimpang dari normal
X_normal = X[y == 0]
print(f"Training pada {len(X_normal):,} transaksi normal...")

iso_forest = IsolationForest(
    n_estimators=200,       # jumlah tree — lebih banyak = lebih stabil
    contamination=0.35,     # estimasi proporsi anomali di data total
    random_state=42,
    n_jobs=-1               # pakai semua CPU core
)
iso_forest.fit(X_normal)

# Generate anomaly score untuk SEMUA data
# score_samples() mengembalikan nilai: semakin negatif = semakin anomali
raw_scores = iso_forest.score_samples(X)

# Normalize ke rentang 0-1: semakin tinggi = semakin NORMAL
# Kita balik polarity: 1 = sangat anomali, 0 = sangat normal
anomaly_score = 1 - ((raw_scores - raw_scores.min()) /
                      (raw_scores.max() - raw_scores.min()))

df['anomaly_score'] = anomaly_score
X['anomaly_score'] = anomaly_score   # masukkan sebagai fitur tambahan

# Evaluasi Isolation Forest
# Karena ini unsupervised, evaluasi dengan membandingkan score vs label asli
iso_pred = (anomaly_score > 0.5).astype(int)   # threshold 0.5
iso_f1   = f1_score(y, iso_pred)
print(f"Isolation Forest F1-Score (threshold 0.5): {iso_f1:.4f}")

# Visualisasi distribusi anomaly score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Distribusi score per label
ax1.hist(anomaly_score[y == 0], bins=40, alpha=0.6,
          color='#3498db', label='Normal (is_at_risk=0)', density=True)
ax1.hist(anomaly_score[y == 1], bins=40, alpha=0.6,
          color='#e74c3c', label='At Risk (is_at_risk=1)', density=True)
ax1.axvline(0.5, color='black', linestyle='--', label='Threshold (0.5)')
ax1.set_title('Distribusi Anomaly Score — Isolation Forest\n'
               'Pemisahan yang jelas = model berhasil membedakan pola')
ax1.set_xlabel('Anomaly Score (0=normal, 1=anomali)')
ax1.legend()

# Plot 2: Score per profil
for p, c in [('normal','#3498db'),('early_stage','#f1c40f'),
              ('escalating','#e67e22'),('heavy_gambler','#e74c3c')]:
    mask = df['profile'] == p
    ax2.hist(anomaly_score[mask], bins=30, alpha=0.55, color=c, label=p, density=True)
ax2.set_title('Distribusi Score per Profil\n'
               'Escalating & heavy harus condong ke kanan (anomali)')
ax2.set_xlabel('Anomaly Score')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('data/isolation_forest_scores.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — XGBOOST CLASSIFIER (SUPERVISED)
# ════════════════════════════════════════════════════════════════════════════
"""
XGBoost adalah gradient boosting yang membangun banyak decision tree
secara berurutan, di mana setiap tree mencoba memperbaiki kesalahan
dari tree sebelumnya.

Kenapa XGBoost untuk data ini?
→ Sangat baik untuk tabular data (data tabel seperti milik kita)
→ Bisa handle imbalanced data dengan scale_pos_weight
→ Feature importance langsung tersedia
→ Cepat ditraining bahkan untuk data besar

PENTING: kita tambahkan anomaly_score dari Isolation Forest sebagai
fitur ke-17 — ini adalah "hybrid model" yang lebih kuat dari keduanya secara terpisah.
"""

print("\n" + "=" * 55)
print("  MODEL 2: XGBOOST CLASSIFIER")
print("=" * 55)

# Update FEATURE_COLS dengan anomaly_score
FEATURE_COLS_FINAL = FEATURE_COLS + ['anomaly_score']
X_final = df[FEATURE_COLS_FINAL].fillna(0)

# ── Split data ──────────────────────────────────────────────────────────────
# stratify=y memastikan proporsi label sama di train dan test
# Ini penting untuk data imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # jaga proporsi label di train/test
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Hitung scale_pos_weight ─────────────────────────────────────────────────
# XGBoost butuh tahu seberapa "berat" kelas minoritas
# Formula: jumlah data negatif / jumlah data positif
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── Training ────────────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=300,               # jumlah tree
    max_depth=6,                    # kedalaman tree — cegah overfitting
    learning_rate=0.05,             # langkah kecil = lebih stabil
    scale_pos_weight=scale_pos_weight,  # handle imbalanced data
    subsample=0.8,                  # ambil 80% data per tree — regularisasi
    colsample_bytree=0.8,           # ambil 80% fitur per tree
    eval_metric='aucpr',            # optimasi PR-AUC saat training
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# Fit dengan early stopping menggunakan validation set
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50                      # print progress setiap 50 iterasi
)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUASI MODEL
# ════════════════════════════════════════════════════════════════════════════
"""
MENGAPA PR-AUC DAN F1, BUKAN ACCURACY?

Bayangkan kita punya 1000 transaksi: 610 at_risk, 390 normal.
Kalau model kita selalu prediksi "at_risk", accuracy = 61% — tapi ini tidak berguna!

PR-AUC (Precision-Recall Area Under Curve) mengukur:
- Precision: dari semua yang diprediksi at_risk, berapa yang benar?
- Recall: dari semua yang at_risk, berapa yang berhasil terdeteksi?
- Trade-off ini lebih relevan untuk kasus deteksi risiko

F1-Score = harmonic mean dari Precision dan Recall
→ Nilai tinggi berarti KEDUANYA precision dan recall baik
"""

y_pred      = xgb_model.predict(X_test)
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]   # probabilitas kelas 1

# Hitung metrik
pr_auc = average_precision_score(y_test, y_pred_prob)
f1     = f1_score(y_test, y_pred)
roc    = roc_auc_score(y_test, y_pred_prob)

print(f"\n{'='*55}")
print(f"  HASIL EVALUASI MODEL")
print(f"{'='*55}")
print(f"  PR-AUC   : {pr_auc:.4f}  ← METRIK UTAMA")
print(f"  F1-Score : {f1:.4f}")
print(f"  ROC-AUC  : {roc:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'At Risk'], digits=4))

# Simpan metrik ke file
with open('data/model_metrics.txt', 'w') as f:
    f.write(f"PR-AUC   : {pr_auc:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc:.4f}\n")
    f.write(f"\n{classification_report(y_test, y_pred, target_names=['Normal','At Risk'])}")

# ── Visualisasi evaluasi ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Evaluasi Model XGBoost — JudolGuard', fontsize=13, fontweight='bold')

# Plot 1: Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
axes[0].plot(recall, precision, color='#e74c3c', linewidth=2,
              label=f'PR-AUC = {pr_auc:.4f}')
axes[0].axhline(y_test.mean(), color='gray', linestyle='--',
                 label=f'Baseline = {y_test.mean():.4f}')
axes[0].fill_between(recall, precision, alpha=0.1, color='#e74c3c')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve\n(Area di bawah kurva = PR-AUC)')
axes[0].legend()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
             xticklabels=['Normal', 'At Risk'],
             yticklabels=['Normal', 'At Risk'])
axes[1].set_title('Confusion Matrix\n(Diagonal = prediksi benar)')
axes[1].set_ylabel('Aktual')
axes[1].set_xlabel('Prediksi')

# Plot 3: Feature Importance
feat_importance = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURE_COLS_FINAL
).sort_values(ascending=True).tail(12)

colors_fi = ['#e74c3c' if 'shift' in f or 'night' in f or 'burst' in f
              else '#3498db' for f in feat_importance.index]
feat_importance.plot(kind='barh', ax=axes[2], color=colors_fi, edgecolor='white')
axes[2].set_title('Feature Importance (Top 12)\nMerah = fitur behavioral kunci')
axes[2].set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('data/model_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CROSS VALIDATION
# ════════════════════════════════════════════════════════════════════════════
"""
Cross validation memastikan performa model bukan kebetulan dari satu split.
Kita bagi data jadi 5 bagian, train 5 kali dengan kombinasi berbeda,
dan rata-rata hasilnya. Ini lebih reliable dari single split.
"""

print("\n Menjalankan 5-Fold Cross Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_model, X_final, y, cv=cv,
                              scoring='average_precision', n_jobs=-1)

print(f"  CV PR-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Scores per fold: {[f'{s:.4f}' for s in cv_scores]}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GENERATE RISK SCORES PER AKUN
# ════════════════════════════════════════════════════════════════════════════
"""
Dashboard butuh risk score PER AKUN, bukan per transaksi.

Cara agregasi:
→ Ambil probabilitas maksimum dari semua transaksi milik satu akun
→ Ini berarti: "seberapa berisiko PERNAH akun ini terlihat?"
→ Lebih cocok untuk keputusan intervensi dibanding rata-rata

Risk levels:
→ 0–30   : Low    → Monitor pasif
→ 31–60  : Medium → Notifikasi ke nasabah
→ 61–80  : High   → Batasi nominal transfer
→ 81–100 : Critical → Eskalasi ke tim compliance/OJK
"""

print("\n Generating risk scores per akun...")

# Prediksi probabilitas untuk SEMUA data (bukan hanya test)
df['risk_prob']  = xgb_model.predict_proba(X_final)[:, 1]
df['risk_score'] = (df['risk_prob'] * 100).round(1)

# Agregasi per akun — ambil score tertinggi yang pernah tercatat
def assign_risk_level(score):
    if score <= 30:   return 'Low'
    elif score <= 60: return 'Medium'
    elif score <= 80: return 'High'
    else:             return 'Critical'

def assign_recommendation(level):
    recs = {
        'Low'     : 'Monitor pasif — tidak ada tindakan segera',
        'Medium'  : 'Kirim notifikasi edukasi ke nasabah',
        'High'    : 'Batasi nominal transfer harian, minta konfirmasi',
        'Critical': 'Eskalasi ke tim compliance & flag ke OJK'
    }
    return recs[level]

account_risk = df.groupby('account_id').agg(
    risk_score_max  = ('risk_score', 'max'),
    risk_score_mean = ('risk_score', 'mean'),
    profile         = ('profile', 'first'),
    is_at_risk_true = ('is_at_risk', 'first'),
    n_transactions  = ('step', 'count'),
    # Top trigger features (rata-rata per akun)
    avg_night_ratio = ('night_ratio_7d', 'mean'),
    avg_tx_24h      = ('tx_count_24h', 'mean'),
    avg_unique_recv = ('unique_recv_7d', 'mean'),
    avg_burst_score = ('burst_score', 'mean'),
    avg_temporal_shift = ('temporal_shift', 'mean'),
    avg_qris_ratio  = ('qris_ratio_7d', 'mean'),
).reset_index()

# Gunakan score maksimum sebagai final risk score
account_risk['final_risk_score'] = account_risk['risk_score_max'].round(1)
account_risk['risk_level']       = account_risk['final_risk_score'].apply(assign_risk_level)
account_risk['recommendation']   = account_risk['risk_level'].apply(assign_recommendation)

# Identifikasi top trigger per akun
def get_top_trigger(row):
    triggers = {
        'Aktivitas malam tinggi'    : row['avg_night_ratio'],
        'Frekuensi tinggi'          : row['avg_tx_24h'] / 20,      # normalize
        'Banyak penerima'           : row['avg_unique_recv'] / 10,  # normalize
        'Velocity burst'            : row['avg_burst_score'] / 5,   # normalize
        'Pergeseran ke malam'       : max(row['avg_temporal_shift'], 0),
        'Penggunaan QRIS tinggi'    : row['avg_qris_ratio'],
    }
    top = sorted(triggers.items(), key=lambda x: x[1], reverse=True)[:2]
    return ' & '.join([t[0] for t in top])

account_risk['top_triggers'] = account_risk.apply(get_top_trigger, axis=1)

# Simpan untuk dashboard
account_risk.to_csv('data/risk_scores.csv', index=False)

# Summary distribusi risk level
print(f"\n Distribusi Risk Level:")
print(account_risk['risk_level'].value_counts().to_string())
print(f"\n risk_scores.csv tersimpan: {len(account_risk)} akun")

# Visualisasi distribusi risk score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(account_risk['final_risk_score'], bins=30,
          color='#e74c3c', alpha=0.7, edgecolor='white')
ax1.axvline(30, color='green', linestyle='--', label='Low/Medium (30)')
ax1.axvline(60, color='orange', linestyle='--', label='Medium/High (60)')
ax1.axvline(80, color='red', linestyle='--', label='High/Critical (80)')
ax1.set_title('Distribusi Risk Score per Akun')
ax1.set_xlabel('Risk Score (0–100)')
ax1.legend(fontsize=9)

level_counts = account_risk['risk_level'].value_counts()
level_order  = ['Low', 'Medium', 'High', 'Critical']
level_colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c']
ax2.bar(
    [l for l in level_order if l in level_counts.index],
    [level_counts.get(l, 0) for l in level_order],
    color=level_colors[:len([l for l in level_order if l in level_counts.index])],
    edgecolor='white'
)
ax2.set_title('Jumlah Akun per Risk Level')
ax2.set_ylabel('Jumlah Akun')

plt.tight_layout()
plt.savefig('data/risk_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SIMPAN MODEL
# ════════════════════════════════════════════════════════════════════════════

joblib.dump(xgb_model,   'models/xgb_judolguard.pkl')
joblib.dump(iso_forest,  'models/isolation_forest.pkl')

print("\n" + "=" * 55)
print("  FASE 4 SELESAI")
print(f"  PR-AUC   : {pr_auc:.4f}")
print(f"  F1-Score : {f1:.4f}")
print("  Output:")
print("    models/xgb_judolguard.pkl")
print("    models/isolation_forest.pkl")
print("    data/risk_scores.csv")
print("    data/model_evaluation.png")
print("\n  Next: Fase 5 — Azure Integration (05_azure_integration.py)")
print("=" * 55)
