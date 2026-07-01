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
        st.metric(label="Total Dataset Ulasan", value="346 Ulasan")
    with col2:
        st.metric(label="Kelas 1 (Puas / Positif)", value="324 Ulasan", delta="93.70%", delta_color="normal")
    with col3:
        st.metric(label="Kelas 0 (Tidak Puas / Negatif)", value="22 Ulasan", delta="-6.30%", delta_color="inverse")
    
    st.info("💡 **Analisis Imbalance Data:** Grafik di atas menunjukkan terjadinya ketimpangan kelas yang sangat ekstrem pada dataset asli (Hanya terdapat 22 ulasan negatif dari total 349 data). Oleh karena itu, penelitian ini wajib menggunakan pendekatan hibrida **SMOTE** untuk menyeimbangkan kelas saat pelatihan model.")
    st.divider()

    # --- TABEL DAN GRAFIK PERBANDINGAN EKSPERIMEN ---
    st.markdown("### 🏆 Hasil Perbandingan Performa Algoritma")
    
    # Membuat dataframe manual 
    data_eksperimen = {
        'Algoritma': ['Logistic Regression', 'SVM', 'Random Forest'],
        'CV Mean F1-Score': ['76%', '73%', '61%'],
        'CV Accuracy': ['96%', '96%', '95%'],
        'Test F1-Score (Macro)': ['63%', '63%', '48%'],
        'Test Accuracy': ['91%', '91%', '93%']
    }
    df_results = pd.DataFrame(data_eksperimen)

    # Menampilkan tabel hasil eksperimen
    st.dataframe(df_results.set_index('Algoritma'), use_container_width=True)

    # Menampilkan visualisasi bar chart performa model menggunakan fitur bawaan Streamlit
    st.markdown("#### Visualisasi Komparasi Nilai Evaluasi Model Final (*Test F1-Score Macro*)")
    # st.bar_chart(data=df_results, x='Algoritma', y='Test F1-Score (Macro)', use_container_width=True)
    
    st.write("Matriks ini menunjukkan detail pengujian model pada data uji asli (Test Set).")
    
    # --- TATA LETAK ASLI BERDERET DAN DIBERI KETERANGAN ---
    col4, col5 = st.columns(2)
    with col4:
        st.image("word_positif.png", caption="Confusion Matrix - Hasil Eksperimen Google Colab", use_container_width=True)
        st.caption("""
        **☁️ Analisis Word Cloud:** Menampilkan aglomerasi kata kunci yang paling sering muncul pada kelompok ulasan * Puas*. 
        Semakin besar ukuran kata kunci, menandakan faktor tersebut merupakan pemicu utama ketidakpuasan konsumen (misal terkait aspek ukuran atau jahitan celana).
        """)
        
    with col5:
        st.image("word_negatif.png", caption="Word Cloud - Ulasan Negatif", use_container_width=True)
        st.caption("""
        **☁️ Analisis Word Cloud:** Menampilkan aglomerasi kata kunci yang paling sering muncul pada kelompok ulasan *Tidak Puas*. 
        Semakin besar ukuran kata kunci, menandakan faktor tersebut merupakan pemicu utama ketidakpuasan konsumen (misal terkait aspek ukuran atau jahitan celana).
        """)
    
    st.divider()
    st.image("Roc_Auc.png", caption="ROC AUC - Hasil Eksperimen Google Colab", use_container_width=400)
    st.caption("""
    **📈 Analisis Kurva ROC-AUC:** Kurva ini menunjukkan perbandingan tingkat kejelian model dalam memisahkan kelas ulasan positif dan negatif. 
    Hasil visualisasi menunjukkan model **SVM (AUC = 0.7143)** dan **Logistic Regression (AUC = 0.7113)** memiliki performa *Fair Classification* yang sangat kompetitif dan seimbang dalam memetakan polaritas sentimen data riil celana jeans Heikin Apparel.
    """)

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
                    
                    # Mengambil nilai probabilitas masing-masing kelas (index 0 = Tidak Puas, index 1 = Puas)
                    prob_tidak_puas = proba[0] * 100
                    prob_puas = proba[1] * 100

                # --- HASIL KEPUTUSAN PREDIKSI AKHIR ---
                col_p1, col_p2 = st.columns(2)
                if prediction[0] == '1':
                    
                    st.balloons()
                    st.success(f"### Hasil Prediksi: **PELANGGAN PUAS (POSITIF)**")
                    st.info(f"Model meyakini ulasan ini bernilai positif dengan tingkat kepuasan sebesar {prob_puas:.2f}%.")
                    with col_p1:
                        st.metric(label="Tingkat Kepuasan (Puas)", value=f"{prob_puas:.2f}%")
                        st.progress(int(prob_puas))
                    with col_p2:
                        st.metric(label="Tingkat Keluhan (Tidak Puas)", value=f"{prob_tidak_puas:.2f}%")
                        st.progress(int(prob_tidak_puas))
                
                        st.divider()
                else:
                    st.markdown("### Hasil Prediksi: <span style='color:#d9534f'>**PELANGGAN TIDAK PUAS (NEGATIF)**</span>", unsafe_allow_html=True)
                    st.info(f"Model mendeteksi adanya keluhan dengan tingkat keyakinan ketidakpuasan sebesar {prob_tidak_puas:.2f}%.")
                    st.warning("⚠️ **Rekomendasi Operasional:** Ulasan terdeteksi negatif. Sistem menyarankan untuk meneruskan data ini ke tim produksi CV Heikin Apparel guna pengecekan kualitas produk/ukuran.")
                    with col_p1:
                        st.metric(label="Tingkat Kepuasan (Puas)", value=f"{prob_puas:.2f}%")
                        st.progress(int(prob_puas))
                    with col_p2:
                        st.metric(label="Tingkat Keluhan (Tidak Puas)", value=f"{prob_tidak_puas:.2f}%")
                        st.progress(int(prob_tidak_puas))
                    
                        st.divider()
                # --- VISUALISASI PROBABILITAS DUA ARAH ---
                
                
                
                
            else:
                st.error("Gagal melakukan prediksi. File tidak terbaca di backend server.")