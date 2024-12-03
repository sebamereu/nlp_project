import os
import re
import pandas as pd
import spacy
import json

# Caricamento del modello SpaCy per l'analisi linguistica
nlp = spacy.load('en_core_web_sm')

# Lista di parole da rimuovere (blacklist personalizzata)
words_to_remove = {"www", "com", "gov", "https", "http", "org", "net", "edu", "uk", "t"}

# Funzione per il preprocessing
def preprocess_text(text):
    # Convertire tutto in lowercase
    text = text.lower()
    
    # Rimuovere URL
    text = re.sub(r'http\S+|www\S+|[a-zA-Z0-9.-]+\.(com|org|net|gov|edu)', '', text)
    
    # Tokenizzazione e filtraggio con spaCy
    doc = nlp(text)
    processed_tokens = [
        token.text for token in doc 
        if not token.is_stop and not token.is_punct and token.is_alpha and token.text not in words_to_remove
    ]
    
    # Restituire il testo rielaborato come una stringa di token separati da spazi
    return ' '.join(processed_tokens)

# Percorso del file originale
file_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/dataset_en_test.json"

# Caricare il dataset
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Creare un DataFrame
df = pd.DataFrame(data)

# Applicare la funzione di preprocessing al testo
df['processed_text'] = df['text'].apply(preprocess_text)

# Visualizzare un esempio di risultato
print("Esempio di testo preprocessato:")
print(df[['text', 'processed_text']].head())

# Salvare i dati preprocessati in un nuovo file JSON
output_dir = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset"
os.makedirs(output_dir, exist_ok=True)

processed_file = os.path.join(output_dir, "dataset_en_processed.json")
df.to_json(processed_file, orient='records', lines=True, force_ascii=False)

print(f"\nDataset preprocessato salvato in: {processed_file}")
