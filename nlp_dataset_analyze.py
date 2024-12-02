# Importare librerie
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Definizione del percorso del dataset e della cartella dei plot
file_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset/dataset_en_test.json"
plot_dir = 'plot'

# Caricare il dataset
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Creare un DataFrame
df = pd.DataFrame(data)

# Visualizzare le prime righe del dataset
print("Prime righe del dataset:")
print(df.head())

# Analisi della distribuzione delle classi
class_distribution = df['category'].value_counts()
class_percentages = class_distribution / len(df) * 100

# Creare un DataFrame per le statistiche
stats_df = pd.DataFrame({
    'Class': class_distribution.index,
    'Count': class_distribution.values,
    'Percentage': class_percentages.values
})

print("\nDistribuzione delle classi:")
print(stats_df)

# Controllare se la cartella dei plot esiste, altrimenti crearla
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Visualizzazione grafica e salvataggio
plt.figure(figsize=(8, 6))
sns.barplot(x=stats_df['Class'], y=stats_df['Count'], palette='viridis')
plt.title('Distribuzione delle Classi nel Train Set')
plt.xlabel('Classi')
plt.ylabel('Conteggio')

# Percorso per il salvataggio del plot
plot_path = os.path.join(plot_dir, 'class_distribution.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot salvato in: {plot_path}")

plt.show()
