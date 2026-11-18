# Segmentasi Pelanggan Online Retail

Aplikasi web untuk memprediksi segmen pelanggan berdasarkan pola transaksi (RFM) menggunakan K-Means Clustering. Dibuat sebagai bagian dari Praktikum Clustering dengan metodologi CRISP-DM.

**Demo:** https://segmentasi-pelanggan-dimas1.streamlit.app

## Latar Belakang

Toko online memiliki data transaksi historis, tetapi belum memiliki pemetaan karakteristik pelanggan. Dengan segmentasi, pengelola toko dapat mengenali pelanggan potensial, aktif, bernilai tinggi, maupun yang mulai tidak aktif, sehingga strategi pemasaran bisa lebih tepat sasaran.

## Dataset

[Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (UCI Machine Learning Repository): transaksi 1 Desember 2010 sampai 9 Desember 2011. Unit analisis adalah pelanggan (`CustomerID`). Nilai Monetary dalam satuan GBP (£).

## Fitur (RFM)

| Fitur | Keterangan |
|---|---|
| Recency | Jumlah hari sejak transaksi terakhir pelanggan |
| Frequency | Jumlah transaksi unik (`InvoiceNo`) per pelanggan |
| Monetary | Total nilai pembelian (`Quantity × UnitPrice`) |

Frequency dan Monetary ditransformasi dengan `log1p` untuk mengurangi skewness, lalu semua fitur distandarisasi dengan `StandardScaler`.

## Model

Algoritma: **K-Means** dengan 3 cluster (`random_state=42`). Evaluasi menggunakan Silhouette Score dan Davies-Bouldin Index, dibandingkan dengan K-Medoids, Hierarchical (Ward), dan DBSCAN pada notebook praktikum.

## Segmen Pelanggan

| Cluster | Segmen | Karakteristik (kira-kira) | Rekomendasi |
|---|---|---|---|
| 0 | Potential Loyalist | Recency ±55 hari, Frequency ±2, Monetary ±£465 | Promo bundling dan program poin untuk mendorong frekuensi |
| 1 | Hibernating | Recency ±256 hari, Frequency ±1, Monetary ±£274 | Kampanye win-back dengan diskon khusus |
| 2 | Champions / VIP | Recency ±30 hari, Frequency ±8, Monetary ±£2.980 | Program loyalitas, akses awal produk, reward eksklusif |

## Cara Menggunakan Aplikasi

1. Isi **Recency**, **Frequency**, dan **Monetary** pelanggan.
2. Klik **Prediksi Cluster**.
3. Aplikasi menampilkan segmen pelanggan dan rekomendasi tindakan pemasaran.

## Struktur Repository

```
├── app.py                                   # Aplikasi Streamlit
├── requirements.txt                         # Dependensi Python
└── deployment_model/
    └── clustering_deployment.joblib         # Model K-Means + scaler
```

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi terbuka di `http://localhost:8501`.

## Catatan

- Hasil segmentasi adalah alat bantu analisis dan perlu divalidasi secara bisnis sebelum dipakai untuk keputusan.
- Nomor cluster K-Means bisa berubah jika model dilatih ulang. Sesuaikan `SEGMENTS` di `app.py` dengan profil cluster yang baru.
- Model dilatih dengan scikit-learn 1.6.1. Memuat dengan versi berbeda dapat menampilkan peringatan versi.

## Teknologi

Python, pandas, NumPy, scikit-learn, joblib, Streamlit.
