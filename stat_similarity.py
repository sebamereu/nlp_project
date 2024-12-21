import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np
import matplotlib.pyplot as plt
import os

def create_combined_dataset(file1_path, file2_path, output_path):
    """
    Unisce due dataset in formato JSON e li salva in un file unico.

    :param file1_path: Percorso al primo file JSON.
    :param file2_path: Percorso al secondo file JSON.
    :param output_path: Percorso del file JSON combinato da salvare.
    """
    # Carica i due dataset
    with open(file1_path, 'r', encoding='utf-8') as file1:
        data1 = json.load(file1)
    
    with open(file2_path, 'r', encoding='utf-8') as file2:
        data2 = json.load(file2)

    # Unisce i due dataset
    combined_data = data1 + data2

    # Salva il dataset combinato
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(combined_data, output_file, ensure_ascii=False, indent=4)
    
    print(f"Dataset combinato salvato in: {output_path}")

def align_datasets_by_size(data1, data2):
    # Riduci il dataset più grande alla dimensione del più piccolo
    target_size = min(len(data1), len(data2))
    data1_reduced = random.sample(data1, target_size)
    data2_reduced = random.sample(data2, target_size)
    return data1_reduced, data2_reduced

def calculate_similarity(file1_path, file2_path):
    # Load JSON files
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    # Align datasets by size
    data1_aligned, data2_aligned = align_datasets_by_size(data1, data2)

    # Extract texts from the JSON data
    texts1 = [entry['text'] for entry in data1_aligned]
    texts2 = [entry['text'] for entry in data2_aligned]

    # Combine texts for vectorization
    combined_texts = texts1 + texts2

    # Convert texts to TF-IDF representation
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # Split TF-IDF matrix into two parts
    tfidf_matrix1 = tfidf_matrix[:len(texts1)]
    tfidf_matrix2 = tfidf_matrix[len(texts1):]

    # Compute cosine similarity between the two sets of texts
    similarity_matrix = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    # Calculate the average similarity
    average_similarity = similarity_matrix.mean()

    return average_similarity

def semantic_similarity(file1_path, file2_path, plot_dir='./results/plot'):
    # Load JSON files
    with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
        data1 = json.load(file1)
        data2 = json.load(file2)

    # Align datasets by size
    data1_aligned, data2_aligned = align_datasets_by_size(data1, data2)

    # Extract texts from the JSON data
    texts1 = [entry['text'] for entry in data1_aligned]
    texts2 = [entry['text'] for entry in data2_aligned]

    # Load pre-trained Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_similarities = util.cos_sim(embeddings1, embeddings2).diagonal()

    # Calculate mean and standard deviation
    mean_similarity = cosine_similarities.mean().item()
    std_dev_similarity = cosine_similarities.std().item()

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "semantic_similarity_distribution.png")

    # Plot and save the distribution
    plt.hist(cosine_similarities.cpu().numpy(), bins=30, alpha=0.7, color='blue')
    plt.title("Distribuzione della Similarità Semantica")
    plt.xlabel("Similarità Coseno")
    plt.ylabel("Frequenza")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot salvato in: {plot_path}")
    return mean_similarity, std_dev_similarity

# Example usage
file1 = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\synth_dataset\synthetic_covid_comments_5000.json"
file2 = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_train.json"

file3 = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_train.json"  # train
output_file = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\synth_dataset\mixed_dataset_2.json"  #mixed dataset

create_combined_dataset(file1, file2, output_file)


# Calculate TF-IDF similarity
average_similarity = calculate_similarity(file1, file2)
print(f"Average Similarity (TF-IDF): {average_similarity:.4f}")

# Calculate semantic similarity and save the plot
mean_sem_sim, std_sem_sim = semantic_similarity(file1, file2)
print(f"Average Semantic Similarity: {mean_sem_sim:.4f}")
print(f"Semantic Similarity Standard Deviation: {std_sem_sim:.4f}")

