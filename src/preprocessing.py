import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

custom_stopwords = set([
    'yang', 'dan', 'atau', 'dengan', 'bahwa', 'untuk', 'pada', 'dalam', 
    'dari', 'serta', 'kepada', 'oleh', 'sebagaimana', 'dimaksud', 'ayat', 
    'pasal', 'huruf', 'angka', 'nomor', 'tahun', 'tentang', 'terkait', 
    'di', 'ini', 'itu', 'tersebut', 'tidak', 'dapat', 'secara', 'lain',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
])

def clean_base(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]
    return ' '.join(words)

def node_preprocessing(input_path, output_path):
    print(f"Membaca data dari: {input_path}")
    df = pd.read_csv(input_path)
    df['Teks_Bersih_BERT'] = df['Teks_Mentah'].apply(clean_base)
    df['Teks_Bersih_BERT'] = df['Teks_Bersih_BERT'].apply(stopword_remover.remove)

    df['Teks_Bersih_LDA'] = df['Teks_Bersih_BERT'].apply(stemmer.stem)
    
    df = df[df['Teks_Bersih_LDA'].str.strip() != '']
    
    df.to_csv(output_path, index=False)
    print(f"Berhasil! Data bersih disimpan di: {output_path}")
    return df