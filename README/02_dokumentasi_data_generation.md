# Synthetic Data Generation & Feature Engineering
## JudolGuard: Early Behavioral Shift Detection System

---

## Mengapa Kami Membuat Data Sendiri?

Tidak ada dataset transaksi judi online yang tersedia secara publik di Indonesia.
Data transaksi nasabah bersifat **sangat tertutup** — bank dan e-wallet tidak mempublikasikannya
karena alasan privasi dan regulasi OJK.

Dataset umum seperti **PaySim** (simulasi fraud mobile money) tidak bisa digunakan langsung
karena dirancang untuk fraud model "hit-and-run" = satu transaksi besar langsung fraud.
Setelah divalidasi melalui EDA, ditemukan bahwa **>90% akun fraud di PaySim adalah burner accounts**
tanpa riwayat perilaku sebelumnya, sehingga tidak bisa mencerminkan behavioral shift bertahap
yang menjadi inti dari solusi JudolGuard.

### Solusi: Azure OpenAI sebagai Data Synthesizer

Kami menggunakan **Azure OpenAI (GPT-4o)** untuk mensintesis data transaksi berdasarkan
karakteristik perilaku judol yang telah didokumentasikan secara resmi oleh PPATK:

> *"Para bandar memecah transaksi dengan nominal yang lebih kecil sehingga frekuensi transaksi
> naik tapi nominal per transaksi menurun"* — PPATK, November 2024

> *"Terjadi pergeseran kanal pembayaran ke QRIS karena membuat alur transaksi sangat cepat
> berpindah dari satu akun ke akun lainnya, menyulitkan penelusuran"* — PPATK, Januari 2025

Pendekatan ini **bukan sekadar workaround** — ini adalah metode yang valid dalam penelitian
keamanan keuangan ketika data sensitif tidak tersedia, selama parameter sintesis
berbasis bukti yang terdokumentasi.

---

## Arsitektur Data: 4 Profil Perilaku

Berbeda dari dataset fraud konvensional yang hanya punya label biner (fraud/tidak),
JudolGuard merancang **4 profil perilaku** yang merepresentasikan spektrum keterlibatan judol:

```
NORMAL ──────► EARLY STAGE ──────► ESCALATING ──────► HEAVY GAMBLER
  │                 │                    │                    │
is_at_risk=0    is_at_risk=0       is_at_risk=1         is_at_risk=1
                                         ▲
                                   TARGET DETEKSI DINI
                                   (Inilah yang sistem
                                    harus tangkap lebih awal)
```

### Profil 1: Normal
- Transaksi 2–4x/hari, jam 07.00–21.00
- Amount konsisten Rp15.000–400.000
- Penerima tetap: merchant, keluarga, utilitas
- **Tidak ada sinyal anomali**

### Profil 2: Early Stage *(baru mulai, belum at-risk)*
- Hari 1–7: masih normal
- Hari 8–14: mulai ada 2–3 transaksi ke "unknown agent", sesekali jam 22–23
- Amount judol masih kecil Rp50.000–150.000
- **Sinyal lemah, belum cukup untuk flag**

### Profil 3: Escalating *(TARGET UTAMA DETEKSI)*
- **Minggu 1–2:** Normal (3–4 tx/hari, siang hari)
- **Minggu 3–4:** Mulai shift (5–8 tx/hari, mulai aktif malam)
- **Minggu 5–6:** Eskalasi penuh (10–15 tx/hari, dominan jam 22.00–04.00, smurfing)
- Ini adalah profil yang paling penting: **perubahan perilaku masih bisa diintervensi**

### Profil 4: Heavy Gambler
- 15–25 tx/hari, mayoritas jam 23.00–05.00
- Full smurfing: semua transfer dipecah Rp10.000–50.000
- 20–30 penerima unik per minggu (money mule network)
- Drain cycle: TOP_UP → habis dalam <2 jam → TOP_UP lagi
- **Sudah sulit diintervensi, tapi penting sebagai sinyal untuk OJK**

---

## Strategi Generasi: Seed + Augmentasi

Karena data transaksi real tidak tersedia dan pemanggilan Azure OpenAI memiliki
batas token, kami menggunakan pendekatan dua tahap:

### Tahap A — Seed Generation via Azure OpenAI
- **20 akun** di-generate langsung dari GPT-4o (5 akun per profil)
- Setiap akun menghasilkan 28–55 transaksi dengan karakteristik profil yang spesifik
- Total: **±798 transaksi seed**
- Azure OpenAI digunakan karena kemampuannya mengikuti instruksi terstruktur
  dan menghasilkan JSON yang konsisten dengan parameter perilaku yang kompleks

### Tahap B — Programmatic Augmentation
- Setiap seed akun menghasilkan **10 variasi** melalui noise realistis:
  - Amount ±20% (mencerminkan variasi kebiasaan belanja)
  - Step offset ±12 jam (zona waktu dan kebiasaan individu berbeda)
  - 10% probabilitas channel swap ewallet → QRIS (untuk profil berisiko)
- Total akhir: **200 akun, ±8.000–12.000 baris transaksi**

> **Mengapa augmentasi valid?**
> Pola perilaku (jam aktif, frekuensi, smurfing) tetap dipertahankan.
> Yang divariasikan hanya parameter minor. Ini setara dengan data preprocessing
> teknik oversampling seperti SMOTE, namun lebih dapat dikontrol dan diinterpretasi.

---

## Feature Engineering: 3 Dimensi Behavioral

Semua fitur dirancang berbasis **tiga dimensi perilaku** yang secara langsung
mencerminkan karakteristik judol yang didokumentasikan PPATK.

---

### Dimensi 1: Temporal — "Kapan dia aktif?"

| Fitur | Cara Hitung | Mengapa Penting |
|---|---|---|
| `hour_of_day` | `step % 24` | Jam spesifik transaksi terjadi |
| `is_night` | 1 jika jam 22–04 | Flag biner aktivitas di luar jam normal |
| `night_ratio_7d` | Rata-rata `is_night` 7 hari terakhir | Seberapa dominan aktivitas malam dalam seminggu |
| `night_ratio_14d` | Rata-rata `is_night` 14 hari terakhir | Baseline lebih panjang untuk deteksi tren |
| `temporal_shift` | `night_ratio_7d` − `night_ratio_prev_7d` | **FITUR KUNCI:** perubahan rasio malam dibanding minggu sebelumnya |

> **`temporal_shift` adalah fitur paling diagnostik.**
> Nilai positif berarti akun mulai bergeser ke aktivitas malam — sinyal paling awal
> dari escalating behavior. Ini yang membedakan JudolGuard dari fraud detection biasa.

---

### Dimensi 2: Velocity — "Seberapa cepat dan berapa banyak?"

| Fitur | Cara Hitung | Mengapa Penting |
|---|---|---|
| `tx_count_24h` | Jumlah transaksi 24 jam terakhir | Deteksi burst activity harian |
| `tx_count_7d` | Jumlah transaksi 7 hari terakhir | Volume mingguan keseluruhan |
| `burst_score` | `tx_count_24h` / (rata-rata harian 7 hari) | Lonjakan frekuensi relatif terhadap baseline |
| `amount_log` | `log(1 + amount)` | Normalisasi amount untuk model |
| `amount_vs_avg_7d` | Amount saat ini / rata-rata amount 7 hari | Deteksi smurfing: amount turun tapi frekuensi naik |
| `total_amount_7d` | Total amount 7 hari | Eksposur keuangan kumulatif |
| `drain_cycle_flag` | 1 jika TOP_UP ≥2 dalam 24 jam + tx ≥6 | Pola habis-isi-habis khas pecandu judol |

> PPATK mendokumentasikan bahwa bandar menginstruksikan pemain untuk memecah deposit
> menjadi nominal kecil ("smurfing"). `amount_vs_avg_7d` yang turun bersamaan dengan
> `tx_count_24h` yang naik adalah tanda smurfing yang paling dapat diandalkan.

---

### Dimensi 3: Multi-Recipient — "Ke mana saja uangnya pergi?"

| Fitur | Cara Hitung | Mengapa Penting |
|---|---|---|
| `unique_recv_7d` | Jumlah penerima unik 7 hari terakhir | Transaksi judol menyebar ke banyak akun |
| `unique_recv_24h` | Jumlah penerima unik 24 jam terakhir | Deteksi burst multi-recipient harian |
| `qris_ratio_7d` | Proporsi transaksi QRIS 7 hari | Pergeseran ke QRIS untuk sembunyikan jejak |
| `dormant_flag` | 1 jika tidak aktif >14 hari lalu tiba-tiba aktif | Pola akun lama yang diaktifkan ulang untuk judol |

---

## Label Target: `is_at_risk`

| Profil | Label | Logika |
|---|---|---|
| normal | 0 | Tidak ada indikasi perilaku berisiko |
| early_stage | 0 | Sinyal terlalu lemah untuk intervensi |
| **escalating** | **1** | **Sedang dalam perubahan — WINDOW INTERVENSI TERBUKA** |
| heavy_gambler | 1 | Sudah dalam perilaku berisiko penuh |

> **Mengapa early_stage = 0?**
> Karena sistem ini dirancang sebagai **intervensi yang tepat sasaran**, bukan surveillance
> massal. Memflag terlalu dini (early_stage) akan menghasilkan false positive tinggi
> dan mengganggu pengguna yang mungkin hanya bertransaksi sesekali ke merchant baru.
> Window intervensi yang optimal adalah saat pola sudah cukup terbentuk (escalating)
> namun belum mencapai kerugian besar (heavy_gambler).

---

## Output Fase Ini

| File | Isi |
|---|---|
| `data/seed_accounts.csv` | 20 akun mentah hasil Azure OpenAI, 798 transaksi |
| `data/judolguard_.csv` | Dataset final: 200 akun, ±8.000–12.000 baris, 22 fitur |
| `data/eda_behavioral_shift.png` | 6 chart bukti behavioral shift antar profil |

---
2. **XGBoost Classifier** — supervised classification dengan `is_at_risk` sebagai target
3. Evaluasi dengan **PR-AUC dan F1-Score** (bukan accuracy — data imbalanced)
