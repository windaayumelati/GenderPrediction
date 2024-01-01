import streamlit as st

def app():
    # HTML untuk mengatur judul di tengah halaman
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 2.8em;'>Aplikasi Prediksi Gender Berdasarkan Ciri dan Karakteristik Wajah Menggunakan KNeighborsClassifier</h1>
        </div>
    """, unsafe_allow_html=True)

    # Menampilkan gambar di tengah halaman
    st.image("gb3.png", use_column_width=True, output_format="JPEG")
