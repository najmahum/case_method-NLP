import fitz
import pandas as pd
import re

def pojk_ke_df(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error membuka PDF: {e}")
        return None

    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc.load_page(page_num).get_text("text") + "\n"

    text = re.sub(r'-\s*\d+\s*-', '\n', full_text)
    text = "\n" + text 
    
    pola_pasal = re.compile(r'\n\s*(Pasal\s+(\d+))\s*\n')
    matches = list(pola_pasal.finditer(text))
    
    data = []
    target_pasal = 1 
    valid_pasal_matches = []
    
    for match in matches:
        nomor_pasal = int(match.group(2))
        
        if nomor_pasal == target_pasal:
            valid_pasal_matches.append(match)
            target_pasal += 1
            
    for i in range(len(valid_pasal_matches)):
        match_sekarang = valid_pasal_matches[i]
        nama_pasal = match_sekarang.group(1).strip()
        start_idx = match_sekarang.end()
        
        if i + 1 < len(valid_pasal_matches):
            end_idx = valid_pasal_matches[i+1].start()
        else:
            end_idx = len(text)
            
        isi_pasal = text[start_idx:end_idx]

        isi_pasal = re.sub(r'\n\s*(BAB\s+[IVXLCDM]+|Bagian\s+[A-Za-z]+|Paragraf\s+\d+).*', '', isi_pasal, flags=re.DOTALL)
        
        isi_pasal = re.sub(r'\n+', ' ', isi_pasal).strip()
        isi_pasal = re.sub(r'\s+', ' ', isi_pasal).strip()
        
        if isi_pasal:
            data.append({
                "Entitas_Hukum": nama_pasal,
                "Teks_Mentah": isi_pasal
            })

    df = pd.DataFrame(data)
    return df