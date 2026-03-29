import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta

# --- KONFIGURASI HALAMAN & TAMPILAN ---
st.set_page_config(page_title="Cek Harga Seledri", page_icon="🌿", layout="centered")

# Custom CSS untuk tampilan lebih modern
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #2e7d32; color: white; border: none; }
    .stButton>button:hover { background-color: #1b5e20; color: white; }
    .result-card { background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; }
    .recommendation { padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32; background-color: #e8f5e9; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 Cek Prediksi Harga Seledri")
st.write("Cari tahu harga jual seledri di masa depan dengan mudah.")

# --- SIDEBAR: UPLOAD DATA ---
with st.sidebar:
    st.header("⚙️ Pengaturan")
    uploaded_file = st.file_uploader("Upload Data Harga (Excel)", type=["xlsx"])
    st.info("Silakan unggah data harga historis terbaru untuk mendapatkan hasil yang akurat.")

# --- FUNGSI PROSES DATA & MODEL ---
def get_prediction_model(df):
    # Cleaning sederhana
    if 'Tanggal,Harga' in df.columns:
        df[['Tanggal', 'Harga']] = df['Tanggal,Harga'].str.split(',', expand=True)
    elif 'harga' in df.columns:
        df.rename(columns={'harga': 'Harga'}, inplace=True)
    
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Harga'] = df['Harga'].astype(int)
    df = df.sort_values('Tanggal').set_index('Tanggal')
    
    # Model SARIMA (Parameter tetap untuk user awam)
    model = SARIMAX(df['Harga'], order=(1,1,1), seasonal_order=(1,1,1,7))
    return model.fit(disp=False), df

# --- HALAMAN UTAMA ---
if uploaded_file:
    fit_model, df_original = get_prediction_model(pd.read_excel(uploaded_file))
    last_date = df_original.index[-1].date()
    current_price = df_original['Harga'].iloc[-1]

    # Pilih Mode Input (Mutually Exclusive)
    mode = st.radio("Pilih cara cek:", 
                    ["Cek Harga Berdasarkan Tanggal", "Cek Tanggal Berdasarkan Target Harga"],
                    horizontal=True)

    st.divider()

    if mode == "Cek Harga Berdasarkan Tanggal":
        target_date = st.date_input("Pilih tanggal yang ingin diprediksi:", 
                                     min_value=last_date + timedelta(days=1),
                                     max_value=last_date + timedelta(days=90))
        
        if st.button("Lihat Prediksi Harga"):
            days_diff = (target_date - last_date).days
            forecast = fit_model.forecast(steps=days_diff)
            predicted_price = int(round(forecast.iloc[-1]))

            # Tampilan Hasil
            st.markdown(f"""
                <div class="result-card">
                    <p style="color: gray; margin-bottom: 0;">Prediksi Harga pada {target_date.strftime('%d %B %Y')}</p>
                    <h1 style="color: #2e7d32; margin-top: 0;">Rp {predicted_price:,}</h1>
                </div>
            """, unsafe_allow_html=True)

            # Rekomendasi
            st.subheader("💡 Rekomendasi")
            if predicted_price > current_price * 1.05:
                st.markdown('<div class="recommendation">🚀 Harga cenderung NAIK. Jika memungkinkan, tunggu tanggal tersebut untuk menjual hasil panen Anda agar untung maksimal.</div>', unsafe_allow_html=True)
            elif predicted_price < current_price * 0.95:
                st.markdown('<div class="recommendation" style="border-left-color: #c62828; background-color: #ffebee;">⚠️ Harga cenderung TURUN. Sebaiknya pertimbangkan untuk menjual lebih awal atau cari pembeli dengan kontrak harga tetap.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="recommendation" style="border-left-color: #f9a825; background-color: #fff9c4;">⚖️ Harga cenderung STABIL. Anda bisa menjual kapan saja sesuai kesiapan stok.</div>', unsafe_allow_html=True)

    else:
        target_price = st.number_input("Masukkan harga yang Anda inginkan (Rp):", min_value=1000, step=500)
        
        if st.button("Cari Tanggal Terbaik"):
            # Prediksi 120 hari ke depan untuk mencari harga
            forecast_long = fit_model.forecast(steps=120)
            # Cari hari pertama yang mendekati atau melebihi target price
            found_date = None
            for i, price in enumerate(forecast_long):
                if price >= target_price:
                    found_date = last_date + timedelta(days=i+1)
                    break
            
            if found_date:
                st.markdown(f"""
                    <div class="result-card">
                        <p style="color: gray; margin-bottom: 0;">Harga Rp {target_price:,} diprediksi akan tercapai pada</p>
                        <h1 style="color: #1976d2; margin-top: 0;">{found_date.strftime('%d %B %Y')}</h1>
                    </div>
                """, unsafe_allow_html=True)
                st.success(f"Tips: Persiapkan stok Anda sekitar 1-2 hari sebelum tanggal tersebut.")
            else:
                st.warning(f"Dalam 4 bulan ke depan, harga diprediksi belum mencapai Rp {target_price:,}. Coba masukkan target harga yang lebih rendah.")

    # --- GRAFIK UNTUK VISUALISASI ---
    st.divider()
    st.subheader("📈 Grafik Tren Harga")
    
    # Forecast 30 hari untuk grafik
    plot_forecast = fit_model.forecast(steps=30)
    plot_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

    fig = go.Figure()
    # Data Asli (30 hari terakhir)
    fig.add_trace(go.Scatter(x=df_original.index[-30:], y=df_original['Harga'][-30:], 
                             mode='lines+markers', name='Harga Terakhir', line=dict(color='#2e7d32')))
    # Data Prediksi
    fig.add_trace(go.Scatter(x=plot_dates, y=plot_forecast, 
                             mode='lines', name='Prediksi Masa Depan', line=dict(dash='dash', color='#ff9800')))

    fig.update_layout(hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 50px; border: 2px dashed #ccc; border-radius: 20px;">
            <h3>Selamat Datang!</h3>
            <p>Silakan upload file Excel harga seledri di menu samping untuk memulai analisis.</p>
        </div>
    """, unsafe_allow_html=True)
