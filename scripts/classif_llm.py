import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
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
from datasets import Dataset, DatasetDict
import torch
# DistilBERT integration
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Paths to JSON files
TRAIN_JSON_PATH = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_train_processed.json"
TEST_JSON_PATH = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_test_processed.json"
PLOT_DIR = r"C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\plot_llm"

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


# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='macro')
    mcc = matthews_corrcoef(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    return {"f1_macro": f1, "mcc": mcc, "confusion_matrix": cm}
  
# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

# Load datasets
train_texts, train_labels = load_json_dataset(TRAIN_JSON_PATH)
test_texts, test_labels = load_json_dataset(TEST_JSON_PATH)

# Encode labels
unique_labels = list(set(train_labels))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
train_labels_encoded = [label_to_index[label] for label in train_labels]
test_labels_encoded = [label_to_index[label] for label in test_labels]

# Classic classifiers and vectorizers
vectorizers = {
    'CountVectorizer': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB()
}

# Results dictionary to store metrics
results = {
    'CountVectorizer': {},
    'TF-IDF': {},
    'DistilBERT': {}
}

# Evaluate classic classifiers
for vec_name, vectorizer in vectorizers.items():
    for clf_name, classifier in classifiers.items():
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        pipeline.fit(train_texts, train_labels_encoded)
        predictions = pipeline.predict(test_texts)
        f1_macro = f1_score(test_labels_encoded, predictions, average='macro')
        mcc = matthews_corrcoef(test_labels_encoded, predictions)
        cm = confusion_matrix(test_labels_encoded, predictions)
        results[vec_name][clf_name] = {
            "F1-macro": f1_macro,
            "MCC": mcc,
            "Confusion Matrix": cm.tolist()
        }



data = DatasetDict({
    'train': Dataset.from_dict({'text': train_texts, 'label': train_labels_encoded}),
    'test': Dataset.from_dict({'text': test_texts, 'label': test_labels_encoded})
})

# Load DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(unique_labels))


data = data.map(tokenize_function, batched=True)
data = data.remove_columns(["text"])
data = data.rename_column("label", "labels")
data.set_format("torch")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data['train'],
    eval_dataset=data['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()

eval_results = trainer.evaluate()
results['DistilBERT'] = {
    "F1-macro": eval_results['eval_f1_macro'],
    "MCC": eval_results['eval_mcc'],
    "Confusion Matrix": eval_results['eval_confusion_matrix']
}

# Save confusion matrices and plots
for method, classifiers_results in results.items():
    for clf_name, metrics in classifiers_results.items():
        cm = metrics["Confusion Matrix"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
        plt.title(f"Confusion Matrix - {clf_name} ({method})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{clf_name.lower().replace(' ', '_')}_{method.lower()}_confusion_matrix.png"))
        plt.close()

# Compare F1-macro scores
plt.figure(figsize=(12, 6))
methods = list(results.keys())
x = np.arange(len(methods))
width = 0.35

f1_scores = [
    np.mean([results[method][clf_name]['F1-macro'] for clf_name in results[method]])
    for method in methods
]

plt.bar(x, f1_scores, width, label='F1-macro')
plt.xlabel('Methods')
plt.ylabel('Average F1-macro')
plt.title('Comparison of Average F1-macro Across Methods')
plt.xticks(x, methods, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'f1_macro_comparison.png'))
plt.close()

print("Results and plots saved.")
