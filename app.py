import joblib, numpy as np, pandas as pd, streamlit as st


st.set_page_config(page_title="Segmentasi Pelanggan", page_icon="🛍️")

@st.cache_resource
def load():
    return joblib.load("deployment_model/clustering_deployment.joblib")

pkg = load()
model, scaler, cols = pkg["model"], pkg["scaler"], pkg["feature_columns"]

# SESUAIKAN dengan hasil profil cluster di notebook Anda
SEGMENTS = {
    0: "Potential Loyalist (belanja moderat, cukup aktif)",
    1: "Hibernating (sangat pasif, risiko churn tinggi)",
    2: "Champions / VIP (baru, sering, nilai tinggi)",
}
ACTIONS = {
    0: "Beri promo bundling dan program poin untuk mendorong frekuensi.",
    1: "Kirim kampanye win-back dengan diskon khusus.",
    2: "Tawarkan program loyalitas, akses awal produk, dan reward eksklusif.",
}

st.title("Segmentasi Pelanggan Online Retail")
st.caption("Prediksi segmen pelanggan dengan K-Means berbasis RFM")

recency = st.number_input("Recency (hari sejak transaksi terakhir)", min_value=0, value=15)
frequency = st.number_input("Frequency (jumlah transaksi)", min_value=1, value=8)
monetary = st.number_input("Monetary (total belanja)", min_value=0.0, value=2500.0)

if st.button("Prediksi Cluster"):
    X = pd.DataFrame({"Recency": [recency],
                      "Frequency_log": [np.log1p(frequency)],
                      "Monetary_log": [np.log1p(monetary)]})[cols]
    c = int(model.predict(scaler.transform(X))[0])
    st.success(f"Cluster {c}: {SEGMENTS.get(c, '-')}")
    st.info(ACTIONS.get(c, ""))
