import streamlit as st
from web_functions import load_data, predict

# Load data
df, x, y = load_data()

def app(df, x, y):
    st.title("Halaman Prediksi Gender")
    st.image('gb4.png', width=280)  # Sesuaikan lebar gambar sesuai kebutuhan

    col1, col2 = st.beta_columns(2)

    with col1:
        long_hair_options = {0: 'No', 1: 'Yes'}
        long_hair = st.selectbox('Apakah Rambut Panjang?', options=list(long_hair_options.keys()), format_func=lambda x: long_hair_options[x], key="long_hair")

        forehead_width_cm = st.number_input('Input Lebar Dahi (cm)', min_value=11.4, max_value=15.5)
        forehead_height_cm = st.number_input('Input Panjang Dahi (cm)', min_value=5.1, max_value=7.1)
        nose_wide_options = {0: 'No', 1: 'Yes'}
        nose_wide = st.selectbox('Apakah Hidung Lebar?', options=list(nose_wide_options.keys()), format_func=lambda x: nose_wide_options[x], key="nose_wide")

    with col2:
        nose_long_options = {0: 'No', 1: 'Yes'}
        nose_long = st.selectbox('Apakah Hidung Panjang?', options=list(nose_long_options.keys()), format_func=lambda x: nose_long_options[x], key="nose_long")
        
        lips_thin_options = {0: 'No', 1: 'Yes'}
        lips_thin = st.selectbox('Apakah Bibir Tipis?', options=list(lips_thin_options.keys()), format_func=lambda x: lips_thin_options[x], key="lips_thin")

        distance_nose_to_lip_long_options = {0: 'No', 1: 'Yes'}
        distance_nose_to_lip_long = st.selectbox('Apakah Jarak Hidung ke Bibir Panjang?', options=list(distance_nose_to_lip_long_options.keys()), format_func=lambda x: distance_nose_to_lip_long_options[x], key="distance_nose_to_lip_long")

    features = [long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]

    # Tombol prediksi
    if st.button("Prediksi Gender"):
        prediction, score = predict(x, y, features)
        st.success("Prediksi Sukses!")

        # Menampilkan hasil prediksi
        st.write('Hasil Prediksi Gender:', prediction[0])

        # Menampilkan informasi model
        st.write('Model yang digunakan: KNeighborsClassifier(n_neighbors=3)')
        st.write('Tingkat akurasi model: {:.2%}'.format(score))

# Panggil fungsi app
app(df, x, y)
