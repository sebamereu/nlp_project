import json
from collections import Counter
from nltk import ngrams
import os
import pandas as pd

# Funzione per calcolare e restituire i top n-grams
def get_top_ngrams(texts, n, top_k=10):
    all_ngrams = []
    for text in texts:
        tokens = text.split()  # Assumiamo che i testi preprocessati siano tokenizzati tramite spazi
        all_ngrams.extend(ngrams(tokens, n))  # Generazione degli n-grams
    
    # Conteggio delle frequenze
    ngram_freq = Counter(all_ngrams)
    return ngram_freq.most_common(top_k)

# Percorso del file preprocessato
preprocessed_file = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/dataset_en_processed.json"

# Caricare il dataset preprocessato
with open(preprocessed_file, 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# Creare un DataFrame
df = pd.DataFrame(data)

# Estrarre i testi preprocessati e le categorie
texts = df['processed_text'].values  # Assumiamo che la colonna si chiami 'processed_text'
categories = df['category'].unique()

# Directory di output
output_dir = "ngrams_analysis_preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Analizzare gli n-grams per ciascuna categoria
results = {}
for category in categories:
    print(f"\nAnalizzando gli n-grams per la categoria: {category}")
    category_texts = df[df['category'] == category]['processed_text']
    
    # Calcolare unigrams, bigrams e trigrams
    unigrams = get_top_ngrams(category_texts, n=1, top_k=10)
    bigrams = get_top_ngrams(category_texts, n=2, top_k=10)
    trigrams = get_top_ngrams(category_texts, n=3, top_k=10)
    
    # Salvare i risultati in un dizionario
    results[category] = {
        "unigrams": unigrams,
        "bigrams": bigrams,
        "trigrams": trigrams
    }
    
    # Stampare i risultati
    print("\nTop 10 unigrams:")
    for unigram, freq in unigrams:
        print(f"{' '.join(unigram)}: {freq}")
    
    print("\nTop 10 bigrams:")
    for bigram, freq in bigrams:
        print(f"{' '.join(bigram)}: {freq}")
    
    print("\nTop 10 trigrams:")
    for trigram, freq in trigrams:
        print(f"{' '.join(trigram)}: {freq}")

# Salvare i risultati in un file JSON
output_file = os.path.join(output_dir, "ngrams_results_preprocessed.json")
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(results, out_file, indent=4, ensure_ascii=False)

print(f"\nRisultati salvati in: {output_file}")
