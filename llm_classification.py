import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Caricamento del tokenizer e del modello BERT pre-addestrato
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)  # 2 classi: conspiracy e critical

# Funzione per il preprocessamento dei dati
def preprocess_data(texts, tokenizer, max_length=512):
    # Tokenizzazione con padding e troncamento
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    return encodings

# Inizializza il LabelEncoder al di fuori della classe (oppure all'interno del costruttore della classe)
label_encoder = LabelEncoder()
label_encoder.fit(["CONSPIRACY", "CRITICAL"])  # Mappa le etichette in numeri

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
        # Convertire le etichette in numeri (0 per 'conspiracy', 1 per 'critical')
        self.labels = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]  # La label è già numerica grazie al LabelEncoder
        encoding = preprocess_data([text], self.tokenizer)
        
        # Restituisci un dizionario con i dati tokenizzati e la label numerica
        return {**encoding, 'labels': torch.tensor(label, dtype=torch.long)}

# Caricare il dataset di test (adattato per il tuo caso)
train_file = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/dataset_en_test.json"

# Carica il dataframe dal file JSON
df = pd.read_json(train_file, orient='records', lines=True)  # File di test
X_test = df['text'].values  # Usa la colonna 'text' come input
y_test = df['category'].values  # Usa la colonna 'category' come etichette

# Creazione del dataset di test
test_dataset = TextDataset(X_test, y_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Funzione di valutazione
def evaluate(model, test_loader):
    model.eval()  # Modalità di valutazione
    predictions = []
    true_labels = []
    
    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Passare i dati attraverso il modello
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Ottieni le previsioni come l'indice del massimo logit
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())  # Sposta le predizioni su CPU
            true_labels.extend(labels.cpu().numpy())  # Sposta le etichette su CPU
    
    # Calcolare l'accuratezza
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Valutazione del modello sul dataset di test
accuracy = evaluate(model, test_loader)
print(f'Accuracy on test set: {accuracy:.4f}')
