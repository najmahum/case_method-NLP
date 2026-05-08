import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic

def run_model(df, features):
    results = []
    
    # --- LDA (Statistik) ---
    
    # 1. LDA + BoW
    lda_bow = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_bow.fit(features['bow'][0])
    results.append({"Model": "LDA", "Input": "BoW", "Skor_LogLikelihood": lda_bow.score(features['bow'][0])})
    
    # 2. LDA + TF-IDF
    lda_tfidf = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_tfidf.fit(features['tfidf'][0])
    results.append({"Model": "LDA", "Input": "TF-IDF", "Skor_LogLikelihood": lda_tfidf.score(features['tfidf'][0])})

    # 3. LDA + N-Grams
    lda_ngrams = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_ngrams.fit(features['ngrams'][0])
    results.append({"Model": "LDA", "Input": "N-Grams", "Skor_LogLikelihood": lda_ngrams.score(features['ngrams'][0])})

    # --- BERTopic (Semantik) ---
    # 4. BERTopic + FastText
    topic_model_ft = BERTopic(language="multilingual")
    topic_model_ft.fit_transform(df['Teks_Bersih_BERT'], features['fasttext'])
    results.append({"Model": "BERTopic", "Input": "FastText", "Skor_LogLikelihood": "N/A (Clustering)"})

    # 5. BERTopic + IndoBERT
    topic_model_bert = BERTopic(language="multilingual")
    topic_model_bert.fit_transform(df['Teks_Bersih_BERT'], features['bert'])
    results.append({"Model": "BERTopic", "Input": "IndoBERT", "Skor_LogLikelihood": "N/A (Clustering)"})
    
    models_dict = {
        "lda_bow": lda_bow,
        "lda_tfidf": lda_tfidf, 
        "lda_ngrams": lda_ngrams,
        "bert_fasttext": topic_model_ft,
        "bert_indobert": topic_model_bert
    }
    
    return pd.DataFrame(results), models_dict