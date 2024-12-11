import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_preprocessed_dataset(file_path):
    """
    Load preprocessed dataset with error handling.
    
    Args:
        file_path (str): Path to the preprocessed JSON file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Attempt to load JSON lines file
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        
        df = pd.DataFrame(data)
        
        # Remove unnecessary columns
        columns_to_drop = ['annotations', 'spacy_tokens']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error loading dataset: {e}")
        return None

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataset into train and validation sets with stratification.
    
    Args:
        df (pd.DataFrame): Input dataset
        test_size (float): Proportion of validation set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: Train and validation sets
    """
    # Ensure 'category' column exists
    if 'category' not in df.columns:
        raise ValueError("Dataset must contain a 'category' column")
    
    train_set, val_set = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['category'], 
        random_state=random_state
    )
    
    return train_set, val_set

def main():
    # Paths
    base_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset"
    input_file = os.path.join(base_path, "dataset_en_train_processed.json")
    output_dir = base_path

    # Load preprocessed dataset
    df = load_preprocessed_dataset(input_file)
    if df is None:
        return

    # Split dataset
    train_set, val_set = split_dataset(df)

    # Analyze class distribution
    print("Class Distribution in Train Set:")
    print(train_set['category'].value_counts(normalize=True))

    print("\nClass Distribution in Validation Set:")
    print(val_set['category'].value_counts(normalize=True))

    # Print set sizes
    print(f"\nTrain Set Size: {train_set.shape[0]} samples")
    print(f"Validation Set Size: {val_set.shape[0]} samples")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save split datasets
    train_file = os.path.join(output_dir, "train_set.json")
    val_file = os.path.join(output_dir, "validation_set.json")

    train_set.to_json(train_file, orient='records', lines=True, force_ascii=False)
    val_set.to_json(val_file, orient='records', lines=True, force_ascii=False)

    print(f"\nTrain Set saved to: {train_file}")
    print(f"Validation Set saved to: {val_file}")

if __name__ == "__main__":
    main()