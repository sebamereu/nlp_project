import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import torch

# Assuming `text_data` and `labels` are ready
# Example:
# text_data = ["Sample text 1", "Sample text 2", ...]
# labels = np.array([0, 1, ...])

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings
def generate_bert_embeddings(text_data):
    embeddings = []
    for text in text_data:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return np.array(embeddings)

# Generate embeddings using BERT
print("Generating BERT embeddings...")
combined_embeddings = generate_bert_embeddings(text_data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_embeddings, labels, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Support Vector Machine (SVM)": SVC(random_state=42, probability=True),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    results[model_name] = {"F1-score": f1, "MCC": mcc}
    print(f"{model_name} - F1-score: {f1:.4f}, MCC: {mcc:.4f}")

# Print all results
print("\nComparison of Models:")
for model_name, metrics in results.items():
    print(f"{model_name}: F1-score = {metrics['F1-score']:.4f}, MCC = {metrics['MCC']:.4f}")

# Prepare data for visualization
model_names = list(results.keys())
f1_scores = [metrics["F1-score"] for metrics in results.values()]
mcc_scores = [metrics["MCC"] for metrics in results.values()]

# Create a DataFrame for Seaborn
results_df = pd.DataFrame({
    "Model": model_names,
    "F1-score": f1_scores,
    "MCC": mcc_scores
})

# Melt the DataFrame for easier plotting
results_melted = pd.melt(results_df, id_vars="Model", var_name="Metric", value_name="Score")

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Performance Comparison of Classical Models", fontsize=16)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Score", fontsize=14)
plt.legend(title="Metric", fontsize=12)
plt.tight_layout()
plt.show()
