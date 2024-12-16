import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-learn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score, 
    matthews_corrcoef
)
from sklearn.pipeline import Pipeline

# Paths to JSON files
TRAIN_JSON_PATH = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_train_processed.json"
TEST_JSON_PATH = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_test_processed.json"
PLOT_DIR = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\plot"

# Ensure the plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

def load_json_dataset(file_path):
    """
    Load dataset from a JSON Lines file.
    """
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            texts.append(entry['processed_text'])
            labels.append(entry['category'])
    return texts, labels

def plot_comparison(results, metric='F1-macro'):
    """
    Create a bar plot comparing vectorization techniques across classifiers.
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    vectorizers = list(results.keys())
    classifiers = list(list(results.values())[0].keys())
    
    x = np.arange(len(classifiers))
    width = 0.35
    
    plt.bar(x - width/2, [results['CountVectorizer'][clf][metric] for clf in classifiers], 
            width, label='Bag of Words (BoW)')
    plt.bar(x + width/2, [results['TF-IDF'][clf][metric] for clf in classifiers], 
            width, label='TF-IDF')
    
    plt.xlabel('Classifiers')
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric} Across Vectorization Techniques')
    plt.xticks(x, classifiers, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOT_DIR, f'{metric.lower().replace("-", "_")}_comparison.png'))
    plt.close()

def plot_confusion_matrix(cm, classnames, model_name, vectorizer_name):
    """
    Plot and save confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classnames, 
                yticklabels=classnames)
    plt.title(f'Confusion Matrix - {model_name} ({vectorizer_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'{model_name.lower().replace(" ", "_")}_{vectorizer_name.lower()}_confusion_matrix.png'))
    plt.close()

# Load datasets
train_texts, train_labels = load_json_dataset(TRAIN_JSON_PATH)
test_texts, test_labels = load_json_dataset(TEST_JSON_PATH)

# Vectorization techniques
vectorizers = {
    'CountVectorizer': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

# Classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB()
}

# Results dictionary to store metrics
results = {
    'CountVectorizer': {},
    'TF-IDF': {}
}

# Encode labels
unique_labels = list(set(train_labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
train_labels_encoded = [label_to_index[label] for label in train_labels]
test_labels_encoded = [label_to_index[label] for label in test_labels]

# Comprehensive comparison
for vec_name, vectorizer in vectorizers.items():
    print(f"\n--- {vec_name} Vectorization ---")
    
    for clf_name, classifier in classifiers.items():
        print(f"\nTraining {clf_name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        # Train the classifier
        pipeline.fit(train_texts, train_labels_encoded)
        
        # Predict on test set
        predictions = pipeline.predict(test_texts)
        
        # Calculate metrics
        f1_macro = f1_score(test_labels_encoded, predictions, average='macro')
        f1_class_0 = f1_score(test_labels_encoded, predictions, pos_label=0)
        f1_class_1 = f1_score(test_labels_encoded, predictions, pos_label=1)
        mcc = matthews_corrcoef(test_labels_encoded, predictions)
        cm = confusion_matrix(test_labels_encoded, predictions)
        
        # Store results
        results[vec_name][clf_name] = {
            "F1-macro": f1_macro,
            "F1-critical": f1_class_0,
            "F1-conspiracy": f1_class_1,
            "MCC": mcc,
            "Confusion Matrix": cm.tolist()
        }
        
        # Print results
        print(f"F1-macro: {f1_macro:.3f}")
        print(f"F1-critical: {f1_class_0:.3f}")
        print(f"F1-conspiracy: {f1_class_1:.3f}")
        print(f"MCC: {mcc:.3f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, unique_labels, clf_name, vec_name)

# Plot comparisons
plot_comparison(results, 'F1-macro')
plot_comparison(results, 'MCC')

# Save comprehensive results
with open(os.path.join(PLOT_DIR, "vectorization_comparison_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print(f"\nComprehensive results saved to {os.path.join(PLOT_DIR, 'vectorization_comparison_results.json')}")
print("Comparison plots and confusion matrices saved in the plot directory.")

# Interpretation notes
print("\n--- Vectorization Technique Comparison Notes ---")
print("1. Bag of Words (CountVectorizer):")
print("   - Simple word frequency representation")
print("   - Treats all words equally")
print("   - Good for smaller datasets")
print("\n2. TF-IDF (Term Frequency-Inverse Document Frequency):")
print("   - Considers word importance across the entire corpus")
print("   - Gives higher weight to rare, more meaningful words")
print("   - Often performs better on larger or more complex datasets")
print("\nRecommendation: Compare the generated plots and JSON results to determine the best technique for your specific dataset.")