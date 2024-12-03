import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Caricare i file JSON
train_file = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/train_set.json"
val_file = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/validation_set.json"

# Carica i dataset di train e validation
train_df = pd.read_json(train_file, orient='records', lines=True)
val_df = pd.read_json(val_file, orient='records', lines=True)

# Estrarre i testi e le etichette (assumendo che la colonna di testo sia 'processed_text' e le etichette 'category')
X_train = train_df['text']
y_train = train_df['category']
X_val = val_df['text']
y_val = val_df['category']

# Convertire le classi da stringhe a numeri (conspiracy -> 0, critical -> 1)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# Creazione della rappresentazione Tf-Idf per i testi
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Creazione dei modelli
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

# Funzione per calcolare F1 e MCC
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    return f1, mcc

# Creare una lista per memorizzare i risultati
results = []

# Allenamento e valutazione dei modelli
for model_name, model in models.items():
    print(f"\nAllenando il modello {model_name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Calcolare F1 e MCC per il validation set
    f1, mcc = evaluate_model(model, X_val_tfidf, y_val)
    
    # Aggiungere i risultati alla lista
    results.append({
        'Model': model_name,
        'F1 Score': f1,
        'MCC': mcc
    })

# Creare una tabella con i risultati
results_df = pd.DataFrame(results)

# Visualizzare la tabella
print("\nTabella di confronto dei modelli:")
print(results_df)

# Salvare la tabella in un file CSV (opzionale)
results_df.to_csv('model_comparison_results.csv', index=False)
