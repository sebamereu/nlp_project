import os
import json

def dataset_size(file_path):
    """
    Calcola la dimensione di un dataset JSON.

    Args:
        file_path (str): Percorso al file JSON.

    Returns:
        tuple: Numero di elementi nel dataset e dimensione del file.
    """
    try:
        # Calcola la dimensione del file in byte
        file_size = os.path.getsize(file_path)
        
        # Carica i dati per contare gli elementi
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Verifica se è una lista di elementi
        if isinstance(data, list):
            num_elements = len(data)
        else:
            raise ValueError("Il file JSON non contiene un array di dati.")

        # Ritorna numero di elementi e dimensione del file
        return num_elements, file_size

    except Exception as e:
        print(f"Errore: {e}")
        return None, None

def format_file_size(size_in_bytes):
    """
    Converte la dimensione del file in un formato leggibile (B, KB, MB, GB).

    Args:
        size_in_bytes (int): Dimensione in byte.

    Returns:
        str: Dimensione formattata.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} TB"

# Esempio di utilizzo
file_path = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\synth_dataset\mixed_dataset_2.json"

num_elements, file_size = dataset_size(file_path)
if num_elements is not None:
    print(f"Numero di elementi nel dataset: {num_elements}")
    print(f"Dimensione del file: {format_file_size(file_size)}")
