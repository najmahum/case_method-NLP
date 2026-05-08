# Case Method NLP: Topic Modelling POJK 22/2023
Tugas ini bertujuan untuk membedakan topik-topik utama dalam dokumen POJK 22/2023 tentang Pelindungan Konsumen Jasa Keuangan. Dilakukan perbandingan dua model dengan 5 representasi kata: statistik (LDA) dan semantik (BERTopic).

## Setup Environment
Proyek ini menggunakan uv. Untuk mendapatkan environment yang sesuai lalu jalankan:
`uv sync`

## Aktivasi Environment:
Windows: .venv\Scripts\activate
Mac/Linux: source .venv/bin/activate

## Struktur Project
├── data/                    # Dataset (PDF asli & CSV hasil perbandingan model)
├── src/                     # Source code modular
│   ├── data_ingestion.py    # Ekstraksi PDF ke teks
│   ├── preprocessing.py     # Pembersihan teks
│   ├── representation.py    # BoW, TF-IDF, N-grams & Embeddings
│   └── modeling.py          # Logika LDA & BERTopic
├── main_reports.ipynb       # Notebook utama untuk eksekusi & visualisasi
├── pyproject.toml           # Konfigurasi project & dependencies
└── uv.lock                  # Lockfile untuk reproduksibilitas

## Metodologi
Proyek ini menerapkan beberapa tahap penting:
- Preprocessing: Memisahkan teknik pembersihan untuk model berbasis frekuensi (dengan stemming) dan model berbasis konteks (tanpa stemming).
- Feature Engineering: Membandingkan representasi teks mulai dari Bag of Words, TF-IDF, hingga Contextual Embeddings (IndoBERT).
- Modeling: Melakukan komparasi performa antara LDA (Latent Dirichlet Allocation) dan BERTopic.
- Visualisasi: Menggunakan grafik interaktif untuk memetakan distribusi kata kunci pada tiap topik.