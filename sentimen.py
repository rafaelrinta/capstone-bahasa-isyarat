# Pembuatan Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#Implementasi Model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer

#DBMS
import pymysql

# memeriksa apakah tabel dalam database MySQL tertentu kosong atau tidak
def is_table_empty(table, host='localhost', user='root', password='', database='review'):
    # Membuat koneksi ke database MySQL menggunakan modul PyMySQL
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    
    # Membuat objek kursor yang digunakan untuk mengeksekusi pernyataan SQL
    cursor = connection.cursor()
    
    # Memeriksa apakah tabel kosong atau tidak
    query_check_empty = f"SELECT COUNT(*) FROM {table}"
    cursor.execute(query_check_empty)
    count_result = cursor.fetchone()[0]

    # Menutup objek kursor dan koneksi database
    cursor.close()
    connection.close()

    return count_result == 0

#Implemnetasi Model dengan Data Baru
def read_mysql_table(table, host='localhost', user='root', password='', database='review'):
    # Membuat koneksi ke database MySQL menggunakan modul PyMySQL
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    
    # Membuat objek kursor yang digunakan untuk mengeksekusi pernyataan SQL
    cursor = connection.cursor()
    
    query = f"SELECT * FROM {table}"
    cursor.execute(query)
    result = cursor.fetchall()
    
    # Mengubah hasil ke dataframe pandas
    df = pd.DataFrame(result)
    
    # menetapkan nama-nama kolom pada Pandas DataFrame berdasarkan deskripsi hasil query SQL
    df.columns = [column[0] for column in cursor.description]
    
    # Menutup objek kursor dan koneksi database
    cursor.close()
    connection.close()
    
    return df

table_name = 'input_review'

#  jika tabel tidak kosong, maka kode di jalankan, begitu juga sebaliknya           
if not is_table_empty(table_name):
    df = read_mysql_table(table_name)
    # text preprocessing
    def preprocess_text(content):
        import nltk
        import re
        nltk.download('stopwords')
        nltk.download('punkt')

        # filtering

        text = re.sub(r'\W', ' ', str(content))
        text = re.sub(r'\s+[a-zA-Z]\s+', ' ', content)
        text = re.sub(r'\^[a-zA-Z]\s+', ' ', content)
        text = re.sub(r'\s+', ' ', content, flags=re.I)
        text = re.sub(r'^b\s+', '', content)

        # case folding
        text = text.lower()

        # Tokenisasi
        tokens = word_tokenize(text)

        # Menghapus stopwords
        stop_words = set(stopwords.words('indonesian'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Menggabungkan kembali tokens menjadi kalimat
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    # Melakukan preprocessing pada semua ulasan
    df['preprocessed_text'] = df['review'].apply(preprocess_text)
    ulasan = df['preprocessed_text']
    ulasan

    # path file pickle
    file_path = 'Model/reviews_tfidf.pickle'

    # membaca file pickle
    with open(file_path, 'rb') as file:
        data_train = pickle.load(file)
        
    # pembuatan vector kata
    # mengonversi kumpulan dokumen teks menjadi matriks representasi TF-IDF
    vectorizer = TfidfVectorizer()
    # menghitung nilai TF-IDF untuk setiap kata dalam data latih
    # dan menghasilkan matriks yang dapat digunakan untuk melatih model.
    train_vector = vectorizer.fit_transform(data_train)
    # Menggabungkan setiap kalimat dalam 'ulasan' menjadi satu string menggunakan metode '" ".join(r)'
    reviews2 = [" ".join(r) for r in ulasan]

    # Menggunakan modul pickle untuk membaca model sentimen yang telah disimpan sebelumnya
    load_model = pickle.load(open('Model/sentimen_model.pkl','rb'))

    # menyimpan hasil prediksi dari model untuk setiap input teks.
    result = []

    # melakukan prediksi sentimen menggunakan model yang telah dimuat sebelumnya.
    for test in reviews2:
        # elemen dalam reviews2 diambil dan dimasukkan ke dalam list test_data
        test_data = [str(test)]
        # Menerapkan transformasi TF-IDF ke data uji (test_data).
        test_vector = vectorizer.transform(test_data).toarray()
        # Melakukan prediksi sentimen untuk data uji yang telah diubah menjadi vektor menggunakan model yang telah dimuat.
        pred = load_model.predict(test_vector)
        # Hasil prediksi (pred[0]) ditambahkan ke dalam list result
        result.append(pred[0])
    # mengelola label-label unik.
    unique_labels(result)

    # Menambahkan kolom baru dengan nama 'label' ke dalam DataFrame df.
    df['label'] = result

    def delete_all_data_from_table(table, host='localhost', user='root', password='', database='review'):
        # Membuat koneksi ke database MySQL menggunakan modul PyMySQL
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # Membuat objek kursor yang digunakan untuk mengeksekusi pernyataan SQL
        cursor = connection.cursor()
        
        # menghapus semua baris dari sebuah tabel dalam database
        query = f"DELETE FROM {table}"
        # Mengeksekusi pernyataan SQL DELETE yang telah dibuat sebelumnya
        cursor.execute(query)
        
        # melakukan commit perubahan pada database setelah sebuah operasi database telah berhasil dieksekusi 
        connection.commit()
        
        # Menutup objek kursor dan koneksi database
        cursor.close()
        connection.close()
    # penghapusan semua data dari tabel dengan nama 'input_review'
    delete_all_data_from_table('input_review')

    def insert_df_into_hasil_model(df, host='localhost', user='root', password='', database='review'):
        # Membuat koneksi ke database MySQL menggunakan modul PyMySQL
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        # Membuat objek kursor yang digunakan untuk mengeksekusi pernyataan SQL
        cursor = connection.cursor()

        # Masukkan setiap baris dari DataFrame ke dalam tabel 'hasil_model'
        for index, row in df.iterrows():
            query = "INSERT INTO hasil_model (id_review, nama, tanggal, review, label) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(query, (row['id_review'], row['nama'], row['tanggal'], row['review'], row['label']))

        # melakukan commit perubahan pada database setelah sebuah operasi database telah berhasil dieksekusi 
        connection.commit()

        # Menutup objek kursor dan koneksi database
        cursor.close()
        connection.close()

    # penyisipan data dari suatu Pandas DataFrame (df) ke dalam tabel database dengan nama 'hasil_model'.
    insert_df_into_hasil_model(df)

    table_name = 'hasil_model'
    # Membaca data dari tabel 'hasil_model' dalam database MySQL
    hasil_df = read_mysql_table(table_name)
    # Menyimpan data dari DataFrame hasil_df ke dalam file CSV
    hasil_df.to_csv('hasil_model2.csv')
    # Membaca data dari file CSV 'hasil_model2.csv' menjadi DataFrame 'data'
    data = pd.read_csv('hasil_model2.csv')
else:
    # Membaca data dari file CSV
    data = pd.read_csv('hasil_model2.csv')

data = data[['review', 'label']]

# Menghitung jumlah data dengan label positif, negatif, dan netral
jumlah_positif = len(data[data['label'] == 5])
jumlah_negatif = len(data[data['label'] == 1])
jumlah_netral = len(data[data['label'] == 3])

# Membuat dashboard Streamlit
st.title('Dashboard Analisis Sentimen SVM')

# Membuat kolom-kolom untuk menyusun metrik
col1, col2, col3 = st.columns(3)

# Menampilkan metrik untuk jumlah data dengan label positif, negatif, dan netral
with col1:
    st.metric(label='Positif (5)', value=jumlah_positif)

with col2:
    st.metric(label='Negatif (1)', value=jumlah_negatif)

with col3:
    st.metric(label='Netral (3)', value=jumlah_netral)

# Membuat bar chart dengan warna yang berbeda
fig, ax = plt.subplots()
labels = ['Positif (5)', 'Negatif (1)', 'Netral (3)']
jumlah_data = [jumlah_positif, jumlah_negatif, jumlah_netral]
colors = ['green', 'red', 'gray']
ax.bar(labels, jumlah_data, color=colors)

# menampilkan suatu gambar (plot) dalam halaman web aplikasi.
st.pyplot(fig)