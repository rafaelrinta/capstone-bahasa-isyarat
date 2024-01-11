# Pembuatan Dashboard
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#Implementasi Model
import nltk
import re
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
    # #menyimpan dataframe review (tipe data series pandas)
    data_content = df['review']

    # casefolding
    data_casefolding = data_content.str.lower()
    data_casefolding.head()

    #filtering

    #url
    filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", str(tweet)) for tweet in data_casefolding]
    #cont
    filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
    #punctuatuion
    filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]
    #  hapus #tagger
    filtering_tagger = [re.sub(r'#([^\s]+)', '', tweet) for tweet in filtering_punctuation]
    #numeric
    filtering_numeric = [re.sub(r'\d+', ' ', tweet) for tweet in filtering_tagger]

    # # filtering RT , @ dan #
    # fungsi_clen_rt = lambda x: re.compile('\#').sub('', re.compile('rt @').sub('@', x, count=1).strip())
    # clean = [fungsi_clen_rt for tweet in filtering_numeric]

    # Data teks yang telah difilter kemudian diubah menjadi Pandas Series 
    data_filtering = pd.Series(filtering_numeric)

    # tokenize
    # tweet dalam data_filtering di-tokenisasi menjadi list kata-kata menggunakan tknzr.tokenize(tweet),
    # kemudian disimpan dalam variabel data_tokenize
    tknzr = TweetTokenizer()
    data_tokenize = [tknzr.tokenize(tweet) for tweet in data_filtering]
    data_tokenize

    #slang word
    path_dataslang = open("Data/kamus kata baku-clear.csv")
    # Menggunakan Pandas untuk membaca file CSV tersebut dan menyimpannya dalam DataFrame dataslang
    dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

    # menggantikan kata slang (slang word) dengan kata baku berdasarkan
    # kamus kata baku yang disimpan dalam DataFrame dataslang
    def replaceSlang(word):
      if word in list(dataslang[0]):
        # jika word terdapat dalam kamus, mendapatkan index (indexslang) dari word dalam kolom pertama
        indexslang = list(dataslang[0]).index(word)
        # Menggantikan word dengan kata baku yang sesuai dari kolom kedua (dataslang[1]) DataFrame dataslang
        return dataslang[1][indexslang]
      else:
        return word

    #  menyimpan hasil tokenisasi yang telah dibersihkan dari kata slang.
    data_formal = []
    # iterasi melalui setiap tweet yang telah di-tokenisasi dan disimpan dalam data_tokenize
    for data in data_tokenize:
      # Menggantikan kata slang dalam setiap kata dalam tweet menggunakan fungsi replaceSlang
      data_clean = [replaceSlang(word) for word in data]
      # Menambahkan tweet yang telah dibersihkan dari kata slang ke dalam list data_formal
      data_formal.append(data_clean)
    # Menghitung jumlah tweet yang telah dibersihkan dari kata slang dan disimpan dalam list data_formal
    len_data_formal = len(data_formal)

    nltk.download('stopwords')
    # Mengambil daftar kata-kata stop words dalam bahasa Indonesia dari NLTK 
    default_stop_words = nltk.corpus.stopwords.words('indonesian')
    # Menggunakan set untuk memungkinkan pencarian lebih cepat daripada menggunakan daftar biasa
    stopwords = set(default_stop_words)

    #  menerima dua parameter: line (teks yang akan dihapus stop words-nya) dan
    # stopwords (kumpulan kata-kata stop words).
    def removeStopWords(line, stopwords):
      words = []
      # Melakukan iterasi melalui setiap kata dalam teks yang disimpan dalam list line
      for word in line:  
        # Mengonversi setiap kata ke dalam bentuk string dan menghapus spasi di awal dan akhir kata
        word=str(word)
        word = word.strip()
        # Memeriksa apakah kata tersebut tidak termasuk dalam kata-kata stop words (word not in stopwords),
        # tidak kosong (word != ""),
        # dan tidak sama dengan "&" (word != "&").
        if word not in stopwords and word != "" and word != "&":
          # Jika memenuhi ketiga kondisi tersebut, maka dimasukkan ke dalam list words.
          words.append(word)

      return words
    #  menerapkan fungsi removeStopWords pada setiap teks dalam list data_formal
    reviews = [removeStopWords(line,stopwords) for line in data_formal]

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
    reviews2 = [" ".join(r) for r in reviews]

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
# Menentukan Label, Jumlah Data, dan Warna untuk Setiap Batang
labels = ['Positif (5)', 'Negatif (1)', 'Netral (3)']
jumlah_data = [jumlah_positif, jumlah_negatif, jumlah_netral]
colors = ['green', 'red', 'gray']
ax.bar(labels, jumlah_data, color=colors)

# menampilkan suatu gambar (menggunakan matplotlib) dalam halaman web aplikasi.
st.pyplot(fig)