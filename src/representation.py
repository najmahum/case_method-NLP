import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models import FastText

def node_representation(df):
    print("--- Memulai Ekstraksi Semua Representasi Teks ---")
    
    print("1. Menghitung Bag of Words (BoW)...")
    bow_vec = CountVectorizer()
    bow_matrix = bow_vec.fit_transform(df['Teks_Bersih_LDA'])
    
    print("2. Menghitung TF-IDF...")
    tfidf_vec = TfidfVectorizer(max_df=0.85, min_df=2)
    tfidf_matrix = tfidf_vec.fit_transform(df['Teks_Bersih_LDA'])
    
    print("3. Mengekstrak N-Grams (2-3 kata)...")
    ngram_vec = CountVectorizer(ngram_range=(2, 3), max_features=1000)
    ngram_matrix = ngram_vec.fit_transform(df['Teks_Bersih_LDA'])
    
    print("4. Menghasilkan FastText Embeddings...")
    sentences = [doc.split() for doc in df['Teks_Bersih_LDA']]
    ft_model = FastText(sentences, vector_size=100, window=5, min_count=1, sg=1)
    
    def get_doc_vector(words, model):
        vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)
    
    ft_embeddings = np.array([get_doc_vector(s, ft_model) for s in sentences])
    
    print("5. Menghasilkan IndoBERT Embeddings...")
    bert_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    bert_embeddings = bert_model.encode(df['Teks_Bersih_BERT'].tolist(), show_progress_bar=True)
    
    return {
        'bow': (bow_matrix, bow_vec),
        'tfidf': (tfidf_matrix, tfidf_vec),
        'ngrams': (ngram_matrix, ngram_vec),
        'fasttext': ft_embeddings,
        'bert': bert_embeddings
    }