import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(page_title="HoaxFinder ID", page_icon="🔍", layout="wide")

# 1. Fungsi Preprocessing (Harus sama dengan saat training)
STOPWORDS = set(["yang", "di", "ke", "dari", "ini", "itu", "dan", "atau", "adalah", "untuk", "dengan", "pada", "sebuah", "oleh", "juga", "sudah", "bisa", "bahwa", "tidak", "akan", "ada", "saya", "kami", "mereka", "dia", "dalam", "setelah", "karena", "seperti", "hanya", "banyak", "tersebut", "namun", "tetapi", "setiap"])

class SimpleBM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.corpus_size = len(corpus_tokens)
        self.avgdl = np.mean([len(d) for d in corpus_tokens])
        self.doc_freqs = {}
        self.idf = {}
        self.doc_tokens = corpus_tokens
        for doc in corpus_tokens:
            for word in set(doc):
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query_tokens):
        scores = np.zeros(self.corpus_size)
        for q in query_tokens:
            if q not in self.idf: continue
            q_idf = self.idf[q]
            for i, doc in enumerate(self.doc_tokens):
                f = doc.count(q)
                if f == 0: continue
                Ld = len(doc)
                denominator = f + self.k1 * (1 - self.b + self.b * (Ld / self.avgdl))
                scores[i] += q_idf * (f * (self.k1 + 1) / denominator)
        return scores

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return tokens

# 2. Load Model Assets (Cached agar cepat)
@st.cache_resource
def load_all_assets():
    # Memuat TF-IDF dari file joblib
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    tfidf_matrix = joblib.load('tfidf_matrix.joblib')
    
    # Memuat BM25 dan Data Display dari file pickle
    with open('hoax_search_assets.pkl', 'rb') as f:
        other_assets = pickle.load(f)
        
    # Menggabungkan semuanya ke dalam satu dictionary
    combined_assets = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'bm25_model': other_assets['bm25_model'],
        'data_display': other_assets['data_display']
    }
    return combined_assets

# Panggil fungsi yang baru
assets = load_all_assets()

# --- UI APP ---
st.title("🇮🇩 HoaxFinder ID")
st.markdown("### Mesin Pencari Klarifikasi Berita Hoaks (Kominfo)")
st.divider()

if assets:
    # Sidebar untuk Filter
    st.sidebar.header("Pengaturan")
    model_choice = st.sidebar.radio("Pilih Model Pencarian:", ("TF-IDF (Cosine)", "BM25 (Best Match)"))
    top_k = st.sidebar.slider("Jumlah Hasil:", 5, 20, 10)

    # Input Pencarian
    query = st.text_input("Masukkan kata kunci atau klaim berita:", placeholder="Contoh: bantuan sultan dubai")

    if query:
        # Preprocess query
        query_tokens = clean_text(query)
        query_clean = " ".join(query_tokens)

        if model_choice == "TF-IDF (Cosine)":
            query_vec = assets['tfidf_vectorizer'].transform([query_clean])
            scores = cosine_similarity(query_vec, assets['tfidf_matrix']).flatten()
        else:
            scores = assets['bm25_model'].get_scores(query_tokens)

        # Ambil Top-K Indeks
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Tampilkan Hasil
        st.write(f"Menampilkan {top_k} hasil terbaik untuk: **{query}**")
        
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                with st.container():
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.subheader(f"{i+1}. {assets['data_display'].iloc[idx]['title']}")
                        st.caption(f"Topik: {assets['data_display'].iloc[idx]['topics']} | Skor: {scores[idx]:.4f}")
                    with col2:
                        url = assets['data_display'].iloc[idx]['url']
                        st.link_button("Buka Sumber", url)
                    
                    # Cuplikan teks (200 karakter pertama)
                    excerpt = assets['data_display'].iloc[idx]['body_text'][:300] + "..."
                    st.write(excerpt)
                    st.divider()
            else:
                if i == 0:
                    st.warning("Tidak ditemukan hasil yang relevan.")
                    break
else:
    st.info("Silakan siapkan file 'hoax_search_assets.pkl' untuk memulai.")