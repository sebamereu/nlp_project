import json
import random
import re
import uuid
import spacy
from faker import Faker

class SyntheticDataGenerator:
    def __init__(self, original_sample):
        self.fake = Faker()
        self.original_sample = original_sample
        
        # Precarica modello spacy per tokenizzazione
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Scarica il modello spacy con: python -m spacy download en_core_web_sm")
            raise

    def generate_synthetic_text(self):
        # Liste di parole chiave e temi per generare testi sintetici
        conspiracy_keywords = [
            'deep state', 'conspiracy', 'hidden agenda', 
            'secret plan', 'manipulation', 'cover-up'
        ]
        public_figures = [
            'Elon Musk', 'Donald Trump', 'Joe Biden', 
            'Hillary Clinton', 'Mark Zuckerberg'
        ]
        platforms = ['Twitter', 'Facebook', 'TikTok', 'Instagram']

        # Genera un testo sintetico con tema cospirativo
        text = f"{random.choice(public_figures)} now revealing what {self.fake.verb()} about the {random.choice(conspiracy_keywords)}. " \
               f"The {random.choice(platforms)} platform is part of a massive {random.choice(conspiracy_keywords)}. " \
               f"Hidden connections reveal {self.fake.bs()}. " \
               f"A secret plan is {self.fake.verb()} right before our eyes. " \
               f"{self.fake.catch_phrase()}. {self.fake.url()}"
        
        return text

    def generate_synthetic_annotations(self, text):
        # Simula annotazioni basate su parole chiave del testo
        categories = ['CAMPAIGNER', 'FACILITATOR', 'AGENT']
        annotations = []
        
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG']:
                annotation = {
                    "span_text": ent.text,
                    "category": random.choice(categories),
                    "annotator": "synthetic_label",
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "start_spacy_token": ent.start,
                    "end_spacy_token": ent.end
                }
                annotations.append(annotation)
        
        return annotations

    def generate_synthetic_sample(self):
        # Genera un campione sintetico
        synthetic_text = self.generate_synthetic_text()
        synthetic_annotations = self.generate_synthetic_annotations(synthetic_text)
        
        # Simula i token di Spacy
        spacy_tokens = self.nlp(synthetic_text)
        spacy_tokens_list = [token.text for token in spacy_tokens]
        
        sample = {
            "id": str(uuid.uuid4())[:8],
            "text": synthetic_text,
            "category": "CONSPIRACY",
            "annotations": synthetic_annotations,
            "spacy_tokens": json.dumps(spacy_tokens_list),
            "processed_text": " ".join([token.lemma_.lower() for token in spacy_tokens])
        }
        
        return sample

    def generate_dataset(self, num_samples=10):
        return [self.generate_synthetic_sample() for _ in range(num_samples)]

# Esempio di utilizzo
def main():
    # Carica il sample originale
    with open(r'C:\Users\sebam\Desktop\NLP\progettoMereuNLP\pan-clef-2024-oppositional-main\pan-clef-2024-oppositional-main\dataset\dataset_en_train_processed.json', 'r') as f:        original_sample = json.load(f)

    generator = SyntheticDataGenerator(original_sample)
    synthetic_dataset = generator.generate_dataset(num_samples=50)

    # Salva il dataset sintetico
    with open('synthetic_dataset.json', 'w') as f:
        json.dump(synthetic_dataset, f, indent=2)

    print(f"Generati {len(synthetic_dataset)} campioni sintetici")

if __name__ == "__main__":
    main()