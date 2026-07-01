import streamlit as st
import joblib
import os
import pandas as pd

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
st.set_page_config(
    page_title="Dashboard & Klasifikasi Kepuasan",
    # page_icon="",
    layout="wide" 
)

# BACKEND: Load model pipeline
@st.cache_resource
def load_sentiment_model():
    model_path = "new.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model_pipeline = load_sentiment_model()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("logo.png", width=80)
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Dashboard Analisis Data", "🔍 Simulasi Klasifikasi Ulasan"]
)

# ==========================================
# HALAMAN 1: DASHBOARD ANALISIS DATA
# ==========================================
if menu == "📊 Dashboard Analisis Data":
    st.title("📊 Dashboard Evaluasi & Performa Model")
    st.subheader("Studi Kasus: Celana Jeans Anak - CV Heikin Apparel")
    st.write("Halaman ini menyajikan ringkasan dataset ulasan asli dan rangkuman hasil eksperimen *Machine Learning* yang telah dilakukan.")
    st.divider()

    # --- KPI METRICS (Rangkuman Data Riil) ---
    st.markdown("### 📈 Ringkasan Karakteristik Dataset Asli")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Dataset Ulasan", value="349 Ulasan")
    with col2:
        st.metric(label="Kelas 1 (Puas / Positif)", value="327 Ulasan", delta="93.70%", delta_color="normal")
    with col3:
        st.metric(label="Kelas 0 (Tidak Puas / Negatif)", value="22 Ulasan", delta="-6.30%", delta_color="inverse")
    
    st.info("💡 **Analisis Imbalance Data:** Grafik di atas menunjukkan terjadinya ketimpangan kelas yang sangat ekstrem pada dataset asli (Hanya terdapat 22 ulasan negatif dari total 349 data). Oleh karena itu, penelitian ini wajib menggunakan pendekatan hibrida **Smote** untuk menyeimbangkan kelas saat pelatihan model.")
    st.divider()

    # --- TABEL DAN GRAFIK PERBANDINGAN EKSPERIMEN ---
    st.markdown("### 🏆 Hasil Perbandingan Performa Algoritma")
    
    # Membuat dataframe manual 
    data_eksperimen = {
        'Algoritma': ['Logistic Regression', 'SVM', 'Random Forest'],
        'CV Mean F1-Score': [0.7698, 0.7533, 0.6108],
        'Test Accuracy': [0.9327, 0.8846, 0.9231],
        'Test F1-Score (Macro)': [0.7485, 0.6351, 0.5798]
    }
    df_results = pd.DataFrame(data_eksperimen)

    # Menampilkan tabel hasil eksperimen
    st.dataframe(df_results.set_index('Algoritma'), use_container_width=True)

    # Menampilkan visualisasi bar chart performa model menggunakan fitur bawaan Streamlit
    st.markdown("#### Visualisasi Komparasi Nilai Evaluasi Model Final (*Test F1-Score Macro*)")
    # st.bar_chart(data=df_results, x='Algoritma', y='Test F1-Score (Macro)', use_container_width=True)
    
    st.write("Matriks ini menunjukkan detail pengujian model pada data uji asli (Test Set).")
    
    col4, col5= st.columns(2)
    with col4:
        st.image("word_positif.png", caption="Confusion Matrix - Hasil Eksperimen Google Colab", use_container_width=True)
    with col5:
        st.image("word_negatif.png", caption="Word Cloud - Ulasan Negatif", use_container_width=True)
    
    st.image("Roc_Auc.png", caption="ROC AUC - Hasil Eksperimen Google Colab", use_container_width=400)

# ==========================================
# HALAMAN 2: SIMULASI KLASIFIKASI ULASAN
# ==========================================
elif menu == "🔍 Simulasi Klasifikasi Ulasan":
    st.title("🔍 Aplikasi Klasifikasi Kepuasan Pelanggan")
    st.write("Silakan masukkan teks ulasan pelanggan baru di bawah ini. Sistem backend akan mendeteksi secara otomatis apakah pelanggan tersebut merasa Puas atau Tidak Puas.")
    st.divider()

    user_review = st.text_area(
        label="Masukkan Teks Ulasan Pelanggan:",
        placeholder="Contoh: Ukuran celananya pas banget di anak saya, jahitannya rapi dan bahannya halus...",
        height=150
    )

    if st.button("Analisis Kepuasan", type="primary"):
        if user_review.strip() == "":
            st.warning("Mohon masukkan teks ulasan terlebih dahulu sebelum menekan tombol analisis.")
        else:
            if model_pipeline is not None:
                with st.spinner("Model backend sedang mengekstraksi TF-IDF dan memprediksi data..."):
                    prediction = model_pipeline.predict([user_review])
                    proba = model_pipeline.predict_proba([user_review])[0]
                    confidence = max(proba) * 100

                # st.success("### Hasil Klasifikasi Sentimen:")
                
                if prediction[0] == '1':
                    st.balloons()
                    st.success("### Hasil Prediksi: **PELANGGAN PUAS (POSITIF)**")
                    st.info(f"Tingkat Keyakinan Algoritma: {confidence:.2f}%")
                else:
                    st.markdown("### Hasil Prediksi: <span style='color:red'>**PELANGGAN TIDAK PUAS (NEGATIF)**</span>", unsafe_allow_html=True)
                    st.info(f"Tingkat Keyakinan Algoritma: {confidence:.2f}%")
                    st.warning("⚠️ **Rekomendasi Operasional:** Ulasan terdeteksi negatif. Sistem menyarankan untuk meneruskan data ini ke tim produksi CV Heikin Apparel guna pengecekan kualitas produk/ukuran.")
            else:
                st.error("Gagal melakukan prediksi. File  tidak terbaca di backend server.")

# st.write("prediction:", prediction)
# st.write("prediction[0]:", prediction[0])
# st.write("type:", type(prediction[0]).__name__)
# st.write("repr:", repr(prediction[0]))
# st.write(type(model_pipeline))
# st.write("Prediction Raw:", prediction)
# st.write("Probability:", proba)