from flask import Flask, render_template, Response, redirect, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import random
import pickle
import json
import nltk
import re
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer

import pymysql
nltk.download('punkt')
nltk.download('wordnet')
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# membaca model chatbot dalam bentuk h5
chatbot_model = load_model("chatbot-bahasa-isyarat.h5")
# membaca file json, untuk memahami dan merespons pertanyaan atau pernyataan dari pengguna.
data_file = open("intents.json").read()
# membaca file model pkl (rb=biner), mengonversi teks input menjadi representasi vektor yang dapat dimengerti oleh model.
words = pickle.load(open("words.pkl", "rb"))
# membaca file model pkl, berisi daftar kelas atau intent yang dapat diprediksi oleh chatbot
classes = pickle.load(open("classes.pkl", "rb"))


# mengimpor modul drawing_utils dari pustaka MediaPipe (menggambar landmark atau hasil dari deteksi objek pada citra.)
mp_drawing = mp.solutions.drawing_utils
# mengimpor modul drawing_styles dari pustaka MediaPipe (mencakup warna dan atribut visual lainnya yang dapat disesuaikan saat menampilkan hasil deteksi)
mp_drawing_styles = mp.solutions.drawing_styles
#  mengimpor modul hands dari pustaka MediaPipe. Modul hands menyediakan model dan fungsi untuk mendeteksi tangan dalam citra atau video
mp_hands = mp.solutions.hands

# membaca model deteksi 
asl_model_path = "ASL_ResNet50.h5"
# memuat model yang telah dilatih sebelumnya dari file yang didefinisikan oleh asl_model_path menggunakan Keras
asl_model = tf.keras.models.load_model(asl_model_path)
# cap belum digunakan secara langsung. cap biasanya digunakan untuk merepresentasikan objek yang menangkap video dari sumber tertentu, seperti webcam atau file video.
cap = None
app = Flask(__name__)

# database konfigurasi
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'review',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentimen')
def sentimen():
    return render_template('sentimen.html')

# Fungsi ini memiliki satu parameter, yaitu onehot. setiap elemennya adalah karakter label yang sesuai dengan indeks nilai maksimum dalam array one-hot encoding
def to_label(onehot):
    label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','del','nothing','space']
    # setiap elemennya adalah karakter label yang sesuai dengan indeks nilai maksimum dalam array one-hot encoding
    r = [label[i] for i in onehot.argmax(1)]
    return r

def gen(): 
    # menginisialisasi objek dengan menggunakan kamera utama yang terhubung ke sistem, diwakili oleh indeks 0
    cap = cv2.VideoCapture(0)
    # menyimpan hasil prediksi atau informasi lainnya
    t = ""

    with mp_hands.Hands(
        # Mengatur jumlah tangan maksimum yang akan dideteksi
        max_num_hands= 1,
        # Mengatur kompleksitas model deteksi tangan.
        model_complexity=0,
        # Mengatur tingkat kepercayaan minimum yang diperlukan untuk mendeteksi tangan
        min_detection_confidence=.8,
        # Mengatur tingkat kepercayaan minimum yang diperlukan untuk melacak tangan setelah deteksi.
        min_tracking_confidence=.5) as hands:
        
        # memulai loop while yang akan berjalan selama objek cap terbuka (kamera masih aktif).
        while cap.isOpened():
            success, image = cap.read()
            # Jika tidak berhasil, maka pesan "Ignoring empty camera frame." dicetak dan eksekusi melanjutkan
            # ke iterasi berikutnya menggunakan continue.
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # mengoptimalkan penggunaan memori. Pengaturan writeable ke False memungkinkan NumPy untuk
            # mengelola memori yang digunakan oleh objek image tanpa membuat salinan data, yang dapat mengurangi overhead..
            image.flags.writeable = False
            # mengonversi format warna citra dari BGR (Blue-Green-Red) ke RGB (Red-Green-Blue)
            # agar mudah saat di konversikan
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # mendeklarasikan tiga variabel w, h, dan c untuk menyimpan dimensi citra. 
            # w menyimpan lebar citra, h menyimpan tinggi citra, dan c menyimpan jumlah saluran warna
            w,h,c = image.shape
            # memproses citra tangan yang telah diubah sebelumnya. Hasil deteksi atau analisis tangan
            # disimpan dalam variabel results.
            results = hands.process(image)

            # mengatur properti writeable dari objek image menjadi True
            # memungkinkan penulisan ulang data citra, 
            image.flags.writeable = True 
            #mengonversi format warna citra dari RGB kembali ke BGR 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    ### Landmark:
            #memeriksa apakah objek results memiliki properti hand_rects yang tidak kosong atau tidak-nilai
            #Jika tidak kosong, artinya terdapat hasil deteksi atau informasi mengenai persegi tangan dalam citra.
            if results.hand_rects:
                # Jika hand_rects tidak kosong, maka baris ini memulai loop for untuk setiap objek hand_rect dalam daftar hand_rects
                for hand_rect in results.hand_rects:
                ## Bouding box:
                    #menghitung koordinat dengan mengalikan nilai cen dari objek hand_rect dengan tinggi citra (h) dan lebar(w).
                    x_cen = hand_rect.x_center*h
                    y_cen = hand_rect.y_center*w
                    # menghitung lebar dan tinggi dengan mengalikan nilai scale dari objek hand_rect dengan tinggi citra (h) dan lebarn(w).
                    scale_w = hand_rect.width * h
                    scale_h = hand_rect.height * w
                
                #panggilan fungsi OpenCV untuk menggambar persegi panjang pada citra
                    cv2.rectangle(image, 
                                #tuple yang menyatakan koordinat pojok kiri atas (top-left) dari persegi panjang yang akan digambar.
                                  (int(x_cen - scale_w/2),int( y_cen - scale_h/2)),
                                #tuple yang menyatakan koordinat pojok kanan bawah (bottom-right) dari persegi panjang yang akan digambar. 
                                  (int(x_cen + scale_w/2), int(y_cen + scale_h/2)), 
                                #Ini adalah tuple warna (B, G, R) yang menentukan warna garis persegi panjang
                                  (0, 255, 0), 2)
                    
                    #meng-crop citra dengan menggunakan indeks slicing. 
                    img = image[int(y_cen - scale_w/2): int(y_cen + scale_w/2), int(x_cen - scale_h/2): int( x_cen + scale_h/2)]
                    #memeriksa apakah array hasil cropping (img) memiliki ukuran yang tidak nol
                    if(np.array(img).size != 0):
                        #menyimpan citra yang telah di-crop ke file dengan nama "crop.jpg"
                        cv2.imwrite('crop.jpg', img)
                        # mengubah ukuran citra yang telah di-crop menjadi (224, 224)
                        img = cv2.resize(img,(224,224))
                        #melakukan normalisasi piksel citra dengan membagi setiap nilai piksel oleh 255
                        img = img/255
                        #menambahkan dimensi batch ke citra dengan menggunakan tf.expand_dims
                        img = tf.expand_dims(img, axis=0)
                        #memprediksi label (isyarat bahasa isyarat) dari citra yang telah di-crop menggunakan model asl_model
                        t = to_label(asl_model.predict(img))[0]
                        print(t)

                #menggambar persegi panjang pada bagian atas citra untuk menempatkan label yang diisi warna biru (255, 0, 0).
                cv2.rectangle(image, (0,0), (h,40), (255,0,0),-1)
                # menambahkan teks (label hasil prediksi) ke citra. Koordinat (int(h/2-4),30) menentukan lokasi teks pada citra.
                cv2.putText(image, t, (int(h/2-4),30), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=1, color=(255,255,255),lineType=cv2.LINE_AA)

# menulis citra yang telah diolah ke file dengan nama "demo.jpg"
            cv2.imwrite('demo.jpg', image)
            #menggunakan yield untuk mengirimkan data citra sebagai respons
            yield (b'--image\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=image')

@app.route('/deteksi')
def deteksi():
    return render_template('deteksi.html')

@app.route('/chatbot')
def chatbot():
    return render_template("chatbot.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]

    #Baris ini membuka file "intents.json" dalam mode baca ("r") dan membaca seluruh kontennya menggunakan metode read(). 
    data_file = open("intents.json").read() 
    #Fungsi json.loads() digunakan untuk melakukan parsing string JSON dan mengonversinya menjadi objek Python
    intents = json.loads(data_file)
 
    #Jika pesan (msg) dimulai dengan string 'my name is', maka bagian dalam blok ini akan dieksekusi.
    if msg.startswith('my name is'):
        #mengekstrak nama dari pesan dengan mengambil substring setelah 'my name is'.
        name = msg[11:]
        #memanggil fungsi predict_class untuk mendapatkan prediksi atau kelas terkait dengan pesan.
        ints = predict_class(msg, chatbot_model)
        #memanggil fungsi getResponse untuk mendapatkan respons yang sesuai dengan prediksi.
        res1 = getResponse(ints, intents)
        #menggantikan placeholder "{n}" dalam respons dengan nama yang diekstrak sebelumnya.
        res = res1.replace("{n}", name)

    #jika pesan dimulai dengan string 'hi my name is'. Jika kondisi ini benar, maka blok ini akan dieksekusi.
    elif msg.startswith('hi my name is'):
        # mengekstrak nama dari pesan dengan mengambil substring setelah 'hi my name is'.
        name = msg[14:]
        # memanggil fungsi predict_class untuk mendapatkan prediksi atau kelas terkait dengan pesan.
        ints = predict_class(msg, chatbot_model)
        #memanggil fungsi getResponse untuk mendapatkan respons yang sesuai dengan prediksi.
        res1 = getResponse(ints, intents)
        #menggantikan placeholder "{n}" dalam respons dengan nama yang diekstrak sebelumnya.
        res = res1.replace("{n}", name)

    #Jika pesan tidak dimulai dengan 'my name is' atau 'hi my name is', maka blok ini akan dieksekusi.
    else:
        #memanggil fungsi predict_class untuk mendapatkan prediksi atau kelas terkait dengan pesan.
        ints = predict_class(msg, chatbot_model)
        # memanggil fungsi getResponse untuk mendapatkan respons yang sesuai dengan prediksi.
        res = getResponse(ints, intents)
        #mengembalikan respons yang telah diproses.
    return res

# chat functionalities
#membersihkan dan memproses kalimat teks.
def clean_up_sentence(sentence):
    #menggunakan modul (NLTK) untuk melakukan tokenisasi kalimat. Hasilnya disimpan dalam variabel sentence_words.
    sentence_words = nltk.word_tokenize(sentence)
    #Mengonversi kata menjadi huruf kecil
    #Menggunakan lemmatizer untuk mengembalikan kata ke bentuk dasarnya (lemma).
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    #mengembalikan daftar kata-kata yang telah dibersihkan dan diproses.
    return sentence_words


#membuat representasi "Bag of Words" (BoW) dari suatu kalimat berdasarkan kata-kata yang telah ditentukan
def bow(sentence, words, show_details=True):
    #memanggil fungsi clean_up_sentence untuk membersihkan dan memproses kata-kata dalam kalimat. 
    sentence_words = clean_up_sentence(sentence)
    #membuat list bag yang terdiri dari nol sepanjang jumlah kata dalam words.
    bag = [0] * len(words)
    # memulai loop untuk setiap kata dalam sentence_words.
    for s in sentence_words:
        #memulai loop untuk setiap kata dalam words, dengan enumerate memberikan indeks (i) dan kata (w).
        for i, w in enumerate(words):
            # memeriksa apakah kata dalam sentence_words (s) sama dengan kata dalam words (w).
            if w == s:
                #jika kata ditemukan, maka elemen ke-i dalam list bag diatur menjadi 1
                bag[i] = 1
                #Jika show_details bernilai True, maka akan dicetak informasi bahwa kata tersebut ditemukan dalam BoW.
                if show_details:
                    print("found in bag: %s" % w)
    #mengembalikan representasi BoW dalam bentuk array numpy.
    return np.array(bag)

#memprediksi kelas atau niat (intent) dari suatu kalimat menggunakan model chatbot_model
def predict_class(sentence, chatbot_model):
    #menghasilkan representasi Bag of Words (BoW) dari kalimat sentence menggunakan daftar kata words.
    p = bow(sentence, words, show_details=False)
    #chatbot_model untuk melakukan prediksi terhadap BoW yang telah dihasilkan. 
    #Hasil prediksi (res) kemudian diambil sebagai nilai probabilitas dari setiap kelas.
    res = chatbot_model.predict(np.array([p]))[0]
    #menetapkan ambang batas probabilitas di mana prediksi dianggap valid.
    ERROR_THRESHOLD = 0.25
    #menghasilkan list results yang berisi indeks dan probabilitas setiap kelas yang memiliki probabilitas di atas ambang batas
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    #mengurutkan list results berdasarkan probabilitas dengan urutan menurun (dari yang tertinggi hingga terendah).
    results.sort(key=lambda x: x[1], reverse=True)
    #membuat list return_list yang akan berisi niat (intent) dan probabilitasnya.
    return_list = []
    #melakukan iterasi melalui list results dan menambahkan setiap pasangan niat (intent) dan probabilitas ke dalam list return_list.
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    #mengembalikan list return_list, yang berisi prediksi niat (intent) dan probabilitasnya dalam format yang terstruktur.
    return return_list

#mendapatkan respons atau jawaban yang sesuai berdasarkan prediksi niat (intent) yang diperoleh dari model chatbot.
def getResponse(ints, intents_json):
    # mengambil nilai niat (intent) dari prediksi pertama dalam list ints yang diberikan.
    tag = ints[0]["intent"]
    # mengambil list niat (intents) dari struktur data JSON intents_json.
    list_of_intents = intents_json["intents"]
    #melakukan iterasi melalui list niat (intents) dan mencari niat yang sesuai dengan nilai tag yang ditemukan sebelumnya.
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    #mengembalikan respons yang telah dipilih secara acak
    return result

# Fungsi menambahkan data ke mysql
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Insert data ke tabel
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# memberikan izin kepada permintaan lintas domain untuk berkomunikasi dengan sumber daya
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors_headers)

# Rute ulasan
@app.route('/ulasan')
def ulasan():
    return render_template('ulasan.html')

# Rute submit form ulasan
@app.route('/submit', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
