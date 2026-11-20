import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Prediksi Pembelian E-commerce', page_icon='🛒')


@st.cache_resource
def muat_model():
    model = joblib.load('models/model_final_gradient_boosting.joblib')
    with open('models/meta.json') as f:
        meta = json.load(f)
    return model, meta


model, meta = muat_model()

st.title('Prediksi Jumlah Pembelian (purchases)')
st.write('Masukkan data perilaku pengguna, lalu klik tombol untuk memprediksi jumlah pembelian.')
st.caption('Model: Gradient Boosting. Nilai input dibatasi pada rentang data latih.')

label = {
    'product_views': 'Jumlah tampilan halaman produk',
    'cart_additions': 'Jumlah penambahan ke keranjang',
    'ad_clicks': 'Jumlah klik iklan',
    'discount_rate': 'Tingkat diskon (0-0,5)',
    'user_rating': 'Rata-rata rating (1-5)',
    'time_on_site': 'Rata-rata waktu sesi (menit)',
}

masukan = {}
col1, col2 = st.columns(2)
for i, (nama, (lo, hi)) in enumerate(meta['rentang'].items()):
    kolom = col1 if i % 2 == 0 else col2
    masukan[nama] = kolom.number_input(
        label[nama], min_value=float(lo), max_value=float(hi),
        value=float(round((lo + hi) / 2, 2)),
    )

masukan['device_type'] = st.selectbox('Jenis perangkat', meta['device_type'])

if st.button('Prediksi'):
    baris = pd.DataFrame([masukan])
    baris = pd.get_dummies(baris, columns=['device_type'])
    # samakan kolom dengan data latih (Desktop = acuan, semua dummy 0)
    baris = baris.reindex(columns=meta['kolom_latih'], fill_value=0).astype(float)
    hasil = model.predict(baris)[0]
    st.success(f'Prediksi jumlah pembelian: {hasil:.1f}')
    st.caption('Rata-rata kesalahan model pada data uji sekitar 7,7 pembelian (MAE).')