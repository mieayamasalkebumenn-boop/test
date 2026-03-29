import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta

# --- KONFIGURASI HALAMAN & TAMPILAN ---
st.set_page_config(page_title="Cek Harga Seledri", page_icon="🌿", layout="centered")

st.markdown(\"\"\"
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #2e7d32; color: white; border: none; }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    .result-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; }
    .recommendation { padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32; background-color: #e8f5e9; font-weight: bold; }
    </style>
    \"\"\", unsafe_allow_html=True)

st.title("🌿 Prediksi & Tren Harga Seledri")
st.write("Pantau pergerakan harga seledri historis dan lihat prediksinya di masa depan.")

# --- SIDEBAR: UPLOAD DATA CSV ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Upload Data (data_seledri_clean.csv)", type=["csv"])
    st.info("Unggah file CSV Anda untuk memunculkan grafik dan prediksi.")

# --- FUNGSI PROSES DATA & MODEL ---
@st.cache_data
def load_and_prepare_data(file):
    df = pd.read_csv(file)
    
    # Deteksi jika data masih tergabung dalam satu kolom (berjaga-jaga)
    if 'Tanggal,Harga' in df.columns:
        df[['Tanggal', 'Harga']] = df['Tanggal,Harga'].str.split(',', expand=True)
        df = df.drop(columns=['Tanggal,Harga'])
        
    # Pastikan nama kolom standar
    if 'harga' in df.columns:
        df.rename(columns={'harga': 'Harga'}, inplace=True)
    if 'tanggal' in df.columns:
        df.rename(columns={'tanggal': 'Tanggal'}, inplace=True)
        
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Harga'] = df['Harga'].astype(int)
    df = df.sort_values('Tanggal').set_index('Tanggal')
    return df

@st.cache_resource
def get_prediction_model(df):
    model = SARIMAX(df['Harga'], order=(1,1,1), seasonal_order=(1,1,1,7))
    return model.fit(disp=False)

# --- HALAMAN UTAMA ---
if uploaded_file:
    # Load Data
    df_original = load_and_prepare_data(uploaded_file)
    
    # 1. MENAMPILKAN GRAFIK DATA HISTORIS
    st.subheader("📊 Grafik Harga Seledri Historis")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=df_original.index, y=df_original['Harga'], 
                                  mode='lines', name='Harga Aktual', line=dict(color='#2e7d32')))
    fig_hist.update_layout(hovermode="x unified", template="plotly_white", 
                           margin=dict(l=0, r=0, t=30, b=0), height=350,
                           xaxis_title="Tanggal", yaxis_title="Harga (Rp)")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()
    
    # 2. PROSES PREDIKSI
    st.subheader("🔮 Cek Prediksi Harga")
    with st.spinner('Memproses model peramalan...'):
        fit_model = get_prediction_model(df_original)
        
    last_date = df_original.index[-1].date()
    current_price = df_original['Harga'].iloc[-1]

    # Pilih Mode Input
    mode = st.radio("Pilih informasi yang ingin dicari:", 
                    ["Cek Harga di Tanggal Tertentu", "Cari Tanggal untuk Target Harga"],
                    horizontal=True)

    if mode == "Cek Harga di Tanggal Tertentu":
        target_date = st.date_input("Pilih tanggal:", 
                                     min_value=last_date + timedelta(days=1),
                                     max_value=last_date + timedelta(days=90))
        
        if st.button("Lihat Prediksi"):
            days_diff = (target_date - last_date).days
            forecast = fit_model.forecast(steps=days_diff)
            predicted_price = int(round(forecast.iloc[-1]))

            st.markdown(f\"\"\"
                <div class="result-card">
                    <p style="color: gray; margin-bottom: 0;">Prediksi Harga pada {target_date.strftime('%d %B %Y')}</p>
                    <h1 style="color: #2e7d32; margin-top: 0;">Rp {predicted_price:,}</h1>
                </div>
            \"\"\", unsafe_allow_html=True)

            if predicted_price > current_price * 1.05:
                st.markdown('<div class="recommendation">🚀 Harga cenderung NAIK dibanding harga terakhir. Jika memungkinkan, tahan stok Anda untuk dijual mendekati tanggal ini.</div>', unsafe_allow_html=True)
            elif predicted_price < current_price * 0.95:
                st.markdown('<div class="recommendation" style="border-left-color: #c62828; background-color: #ffebee;">⚠️ Harga cenderung TURUN. Sebaiknya segera jual stok Anda sebelum harga anjlok lebih jauh.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="recommendation" style="border-left-color: #f9a825; background-color: #fff9c4;">⚖️ Harga cenderung STABIL. Lakukan penjualan seperti biasa.</div>', unsafe_allow_html=True)

    else:
        target_price = st.number_input("Masukkan target harga jual (Rp):", min_value=1000, step=500)
        
        if st.button("Cari Tanggal Terbaik"):
            forecast_long = fit_model.forecast(steps=120)
            found_date = None
            for i, price in enumerate(forecast_long):
                if price >= target_price:
                    found_date = last_date + timedelta(days=i+1)
                    break
            
            if found_date:
                st.markdown(f\"\"\"
                    <div class="result-card">
                        <p style="color: gray; margin-bottom: 0;">Harga Rp {target_price:,} diprediksi tercapai pada</p>
                        <h1 style="color: #1976d2; margin-top: 0;">{found_date.strftime('%d %B %Y')}</h1>
                    </div>
                \"\"\", unsafe_allow_html=True)
                st.success("Siapkan logistik dan stok hasil panen Anda beberapa hari sebelum tanggal tersebut.")
            else:
                st.warning(f"Dalam 4 bulan ke depan, harga diprediksi belum mencapai Rp {target_price:,}.")

else:
    st.markdown(\"\"\"
        <div style="text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 20px;">
            <h3 style="color: #555;">Belum Ada Data</h3>
            <p style="color: #777;">Tarik dan lepas (drag & drop) file <b>data_seledri_clean.csv</b> Anda pada panel di sebelah kiri.</p>
        </div>
    \"\"\", unsafe_allow_html=True)
