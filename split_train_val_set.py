# Importare librerie
import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Percorso del file originale
file_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/dataset_en_test.json"

# Caricare il dataset
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Creare un DataFrame
df = pd.DataFrame(data)

# Rimuovere le colonne inutili
df = df.drop(columns=['annotations', 'spacy_tokens'], errors='ignore')

# Dividere il dataset in train e validation set (80% - 20%)
train_set, val_set = train_test_split(df, test_size=0.2, stratify=df['category'], random_state=42)

# Confermare la distribuzione delle classi
print("Distribuzione delle classi nel train set:")
print(train_set['category'].value_counts(normalize=True))

print("\nDistribuzione delle classi nel validation set:")
print(val_set['category'].value_counts(normalize=True))

# Stampare la dimensione dei set
print(f"\nDimensione del train set: {train_set.shape[0]} elementi")
print(f"Dimensione del validation set: {val_set.shape[0]} elementi")

# Salvare i nuovi dataset in file JSON
output_dir = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset"
os.makedirs(output_dir, exist_ok=True)

train_file = os.path.join(output_dir, "train_set.json")
val_file = os.path.join(output_dir, "validation_set.json")

train_set.to_json(train_file, orient='records', lines=True, force_ascii=False)
val_set.to_json(val_file, orient='records', lines=True, force_ascii=False)

print(f"\nTrain set salvato in: {train_file}")
print(f"Validation set salvato in: {val_file}")
