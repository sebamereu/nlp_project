import os
import re
import pandas as pd
import spacy
import json

# Load SpaCy linguistic analysis model
nlp = spacy.load('en_core_web_sm')

# Custom blacklist of words to remove
words_to_remove = {
    "www", "com", "gov", "https", "http", "org", 
    "net", "edu", "uk", "t"
}

def preprocess_text(text):
    """
    Perform comprehensive text preprocessing using SpaCy.
    
    Key preprocessing steps:
    1. Convert text to lowercase
    2. Remove URLs and web-related strings
    3. Tokenize and filter text using SpaCy
    4. Remove stopwords, punctuation, and specified words
    
    Args:
        text (str): Input text to be preprocessed
    
    Returns:
        str: Preprocessed text as a space-separated string of tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and web-related strings
    text = re.sub(
        r'http\S+|www\S+|[a-zA-Z0-9.-]+\.(com|org|net|gov|edu)', 
        '', 
        text
    )
    
    # Tokenize and filter with SpaCy
    doc = nlp(text)
    processed_tokens = [
        token.text for token in doc 
        if (
            not token.is_stop and 
            not token.is_punct and 
            token.is_alpha and 
            token.text not in words_to_remove
        )
    ]
    
    # Return processed text as a space-separated string
    return ' '.join(processed_tokens)

def main():
    # Path to the original dataset
    file_path = (
        "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/"
        "pan-clef-2024-oppositional-main/"
        "pan-clef-2024-oppositional-main/dataset/dataset_en_test.json"
    )

    # Load dataset
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Apply preprocessing to text
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Display preprocessed text example
    print("Example of preprocessed text:")
    print(df[['text', 'processed_text']].head())

    # Prepare output directory
    output_dir = (
        "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/"
        "pan-clef-2024-oppositional-main/"
        "pan-clef-2024-oppositional-main/dataset"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed data
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