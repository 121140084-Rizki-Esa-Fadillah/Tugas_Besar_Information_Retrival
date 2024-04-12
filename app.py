from flask import Flask, render_template, request, send_from_directory
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
import os

# Download data NLTK jika belum diunduh sebelumnya
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Path untuk file CSV dan model Word2Vec
csv_file_path = "imdb_indonesian_movies_2.csv"
word2vec_model_path = "word2vec_model.bin"

# Periksa apakah model Word2Vec sudah ada, jika tidak, lakukan pelatihan
if not os.path.exists(word2vec_model_path):
    print("Training Word2Vec model...")
    dataset = pd.read_csv(csv_file_path, encoding="utf-8")

    # Proses teks dan bersihkan
    def preprocess_text(text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
        return filtered_tokens

    # Buat daftar kalimat yang sudah diproses
    sentences = [preprocess_text(text) for text in dataset['ringkasan_sinopsis']]

    # Buat dan latih model Word2Vec
    Word2Vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Simpan model ke dalam format biner
    Word2Vec_model.save(word2vec_model_path)
    print("Word2Vec model trained and saved.")
else:
    # Jika model sudah ada, langsung muat
    Word2Vec_model = Word2Vec.load(word2vec_model_path)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens

def word_overlap_similarity(query_tokens, synopsis_tokens):
    """
    Menghitung tingkat kemiripan berdasarkan jumlah kata yang sama antara query dan sinopsis.
    """
    overlap_count = len(set(query_tokens) & set(synopsis_tokens))
    return overlap_count

dataset = pd.read_csv(csv_file_path, encoding="utf-8")

@app.route('/')
def index():
    return render_template('index.html')

from operator import itemgetter

@app.route("/search", methods=["POST"])
def search():
    query = request.form['query']
    genre = request.form['genre']

    preprocessed_query = preprocess_text(query)

    results = []

    for index, row in dataset.iterrows():
        preprocessed_synopsis = preprocess_text(row['ringkasan_sinopsis'])
        similarity = word_overlap_similarity(preprocessed_query, preprocessed_synopsis)
        if genre.lower() in row['genre'].lower():
            similarity_percentage = round((similarity / len(preprocessed_query)) * 100, 2)
            if similarity_percentage > 50:
                results.append((row['judul_film'], row['ringkasan_sinopsis'], row['genre'], similarity_percentage))

    # Urutkan hasil berdasarkan kesamaan persentase paling tinggi
    sorted_results = sorted(results, key=itemgetter(3), reverse=True)

    return render_template('result.html', results=sorted_results)

if __name__ == '__main__':
    app.run(debug=True)
