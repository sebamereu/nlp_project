import json
import os
import logging
from typing import List, Tuple
import pandas as pd
from collections import Counter
from nltk import ngrams

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s'
)

def load_preprocessed_dataset(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed dataset with error handling.
    
    Args:
        file_path (str): Path to preprocessed JSON file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None

def get_top_ngrams(
    texts: List[str], 
    n: int, 
    top_k: int = 10
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Calculate top N-grams from a list of texts.
    
    Args:
        texts (List[str]): List of preprocessed texts
        n (int): N-gram size (1 for unigrams, 2 for bigrams, etc.)
        top_k (int): Number of top N-grams to return
    
    Returns:
        List of top N-grams with their frequencies
    """
    all_ngrams = []
    for text in texts:
        tokens = text.split()  # Assumes texts are space-tokenized
        all_ngrams.extend(ngrams(tokens, n))
    
    ngram_freq = Counter(all_ngrams)
    return ngram_freq.most_common(top_k)

def analyze_ngrams(df: pd.DataFrame, dataset_type: str, output_dir: str) -> dict:
    """
    Analyze N-grams for each category in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        dataset_type (str): 'train' or 'validation'
        output_dir (str): Directory to save results
    
    Returns:
        dict: N-gram analysis results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique categories
    categories = df['category'].unique()
    
    results = {}
    for category in categories:
        logging.info(f"Analyzing N-grams for {dataset_type} set, category: {category}")
        
        # Filter texts for current category
        category_texts = df[df['category'] == category]['processed_text']
        
        # Calculate N-grams
        results[category] = {
            "unigrams": get_top_ngrams(category_texts, n=1),
            "bigrams": get_top_ngrams(category_texts, n=2),
            "trigrams": get_top_ngrams(category_texts, n=3)
        }
        
        # Print results
        print(f"\nN-grams for {dataset_type} set, category {category}:")
        for ngram_type, ngram_list in [
            ("Unigrams", results[category]["unigrams"]),
            ("Bigrams", results[category]["bigrams"]),
            ("Trigrams", results[category]["trigrams"])
        ]:
            print(f"\nTop 10 {ngram_type}:")
            for ngram, freq in ngram_list:
                print(f"{' '.join(ngram)}: {freq}")
    
    # Save results to JSON
    output_file = os.path.join(output_dir, f"ngrams_results_{dataset_type}_set.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    logging.info(f"N-grams analysis results for {dataset_type} set saved to: {output_file}")
    
    return results

def main():
    # Paths
    base_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset"
    train_file = os.path.join(base_path, "dataset_en_train_processed.json")
    val_file = os.path.join(base_path, "dataset_en_test_processed.json")
    output_dir = os.path.join(base_path, "results/ngrams_analysis")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load train dataset
    train_df = load_preprocessed_dataset(train_file)
    if train_df is None:
        logging.error("Failed to load training dataset")
        return
    
    # Load validation dataset
    val_df = load_preprocessed_dataset(val_file)
    if val_df is None:
        logging.error("Failed to load validation dataset")
        return
    
    # Analyze N-grams for training set
    analyze_ngrams(train_df, 'train', output_dir)
    
    # Analyze N-grams for validation set
    analyze_ngrams(val_df, 'validation', output_dir)

if __name__ == "__main__":
    main()