import os
import re
import pandas as pd
import spacy
import json
import unicodedata
from nltk.stem import PorterStemmer
import nltk

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

# Load the SpaCy language model for linguistic analysis
nlp = spacy.load('en_core_web_sm')

# Initialize Porter Stemmer
porter_stemmer = PorterStemmer()

# Custom list of additional stopwords (optional)
custom_stopwords = {
    "www", "com", "gov", "https", "http", "org", 
    "net", "edu", "uk", "t"
}

def unicode_to_ascii(text):
    """
    Convert Unicode characters to their closest ASCII representation.
    
    Args:
        text (str): Input text with potential Unicode characters
    
    Returns:
        str: Text with Unicode characters replaced by ASCII equivalents
    """
    # Normalize Unicode characters
    normalized = unicodedata.normalize('NFKD', text)
    
    # Remove non-ASCII characters and decode to ASCII
    ascii_text = normalized.encode('ascii', 'ignore').decode('utf-8')
    
    return ascii_text

def preprocess_text(text):
    """
    Perform comprehensive text preprocessing:
    - Convert Unicode to ASCII
    - Convert to lowercase
    - Remove URLs and special characters
    - Remove stopwords, punctuation, and single-letter words
    - Apply Porter stemming
    
    Args:
        text (str): Input text to preprocess
    
    Returns:
        str: Preprocessed text as a space-separated string of tokens
    """
    # Convert Unicode to ASCII
    text = unicode_to_ascii(text)
    
    # Normalize text: convert to lowercase (already done in unicode_to_ascii)
    text = text.lower()
    
    # Remove URLs and special characters
    text = re.sub(
        r'http\S+|www\S+|[a-zA-Z0-9.-]+\.(com|org|net|gov|edu)', 
        '', 
        text
    )
    
    # Tokenize using SpaCy
    doc = nlp(text)
    
    # Filter and process tokens
    tokens = [
        porter_stemmer.stem(token.text) for token in doc 
        if (
            not token.is_stop and  # Remove stopwords
            not token.is_punct and  # Remove punctuation
            (token.is_alpha or token.is_digit) and  # Keep alphabetic and numeric tokens
            len(token.text) > 1 and  # Remove single-letter words
            token.text not in custom_stopwords  # Remove custom stopwords
        )
    ]
    
    # Return the preprocessed text as a single string
    return ' '.join(tokens)

# Rest of the script remains the same as in the previous version

def preprocess_file(input_file, output_file):
    """
    Preprocess the input dataset and save it to a new file.
    
    Args:
        input_file (str): Path to the input dataset file.
        output_file (str): Path to save the preprocessed dataset.
    """
    # Load the dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format at {input_file}")
        return

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame(data)

    # Preprocess the text column
    if 'text' not in df.columns:
        print(f"Error: 'text' column does not exist in the dataset {input_file}.")
        return
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Save the preprocessed dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_json(
        output_file, 
        orient='records', 
        lines=True, 
        force_ascii=False
    )
    print(f"Preprocessed dataset saved to: {output_file}")

def main():
    base_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/" \
                "pan-clef-2024-oppositional-main/" \
                "pan-clef-2024-oppositional-main/dataset/"

    # Paths for training and test datasets
    train_file = os.path.join(base_path, "dataset_en_train.json")
    train_output = os.path.join(base_path, "dataset_en_train_processed.json")

    test_file = os.path.join(base_path, "dataset_en_test.json")
    test_output = os.path.join(base_path, "dataset_en_test_processed.json")

    # Preprocess training dataset
    print("Processing training dataset...")
    preprocess_file(train_file, train_output)

    # Preprocess test dataset
    print("Processing test dataset...")
    preprocess_file(test_file, test_output)

if __name__ == "__main__":
    main()