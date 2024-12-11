import os
import re
import pandas as pd
import spacy
import json

# Load the SpaCy language model for linguistic analysis
nlp = spacy.load('en_core_web_sm')

# Custom list of additional stopwords (optional)
custom_stopwords = {
    "www", "com", "gov", "https", "http", "org", 
    "net", "edu", "uk", "t"
}

def preprocess_text(text):
    """
    Perform text preprocessing in several steps:
    - Convert text to lowercase
    - Remove punctuation and special characters
    - Tokenize text
    - Remove stopwords and irrelevant tokens
    
    Args:
        text (str): Input text to preprocess
    
    Returns:
        str: Preprocessed text as a space-separated string of tokens
    """
    # Normalize text: convert to lowercase
    text = text.lower()
    
    # Remove URLs and special characters
    text = re.sub(
        r'http\S+|www\S+|[a-zA-Z0-9.-]+\.(com|org|net|gov|edu)|[^a-zA-Z\s]', 
        '', 
        text
    )
    
    # Tokenize and filter using SpaCy
    doc = nlp(text)
    tokens = [
        token.text for token in doc 
        if (
            not token.is_stop and  # Remove stopwords
            not token.is_punct and  # Remove punctuation
            token.is_alpha and  # Keep only alphabetic tokens
            token.text not in custom_stopwords  # Remove custom stopwords
        )
    ]
    
    # Return the preprocessed text as a single string
    return ' '.join(tokens)

def main():
    # Path to the dataset
    file_path = (
        "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/"
        "pan-clef-2024-oppositional-main/"
        "pan-clef-2024-oppositional-main/dataset/dataset_en_train.json"
    )

    # Load the dataset
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format at {file_path}")
        return

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(data)

    # Preprocess the text column
    if 'text' not in df.columns:
        print("Error: 'text' column does not exist in the dataset.")
        return
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Display an example of preprocessed text
    print("Example of preprocessed text:")
    print(df[['text', 'processed_text']].head())

    # Save the preprocessed dataset
    output_dir = (
        "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/"
        "pan-clef-2024-oppositional-main/"
        "pan-clef-2024-oppositional-main/dataset"
    )
    os.makedirs(output_dir, exist_ok=True)

    processed_file = os.path.join(output_dir, "dataset_en_processed.json")
    df.to_json(
        processed_file, 
        orient='records', 
        lines=True, 
        force_ascii=False
    )

    print(f"\nPreprocessed dataset saved to: {processed_file}")

if __name__ == "__main__":
    main()
