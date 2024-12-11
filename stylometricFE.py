import json
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import textstat
import os

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

class StylometricFeatureExtractor:
    def __init__(self):
        # Load spaCy English model for advanced linguistic analysis
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize VADER sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion lexicon (example - can be expanded)
        self.emotion_lexicon = {
            'anger': ['angry', 'furious', 'rage', 'hate', 'threat', 'conspiracy', 'manipulate'],
            'fear': ['afraid', 'scared', 'terror', 'panic', 'danger', 'risk', 'threat'],
            'sadness': ['sad', 'depressed', 'grief', 'sorrow', 'hopeless', 'betrayal'],
            'joy': ['happy', 'joy', 'love', 'excitement', 'hope', 'truth']
        }

    def extract_syntactic_complexity(self, text):
        """
        Extract syntactic complexity features
        """
        doc = self.nlp(text)
        
        # Average dependency tree depth
        def get_depth(token):
            depth = 0
            while token.head != token:
                depth += 1
                token = token.head
            return depth
        
        depths = [get_depth(token) for token in doc]
        
        return {
            'avg_sentence_depth': np.mean(depths) if depths else 0,
            'max_sentence_depth': max(depths) if depths else 0,
            'syntactic_complexity_score': textstat.flesch_reading_ease(text),
            'avg_parse_tree_depth': np.mean(depths) if depths else 0
        }

    def extract_sentiment(self, text):
        """
        Comprehensive sentiment analysis
        """
        # VADER sentiment scores
        vader_scores = self.sia.polarity_scores(text)
        
        return {
            'sentiment_positive': vader_scores['pos'],
            'sentiment_negative': vader_scores['neg'],
            'sentiment_neutral': vader_scores['neu'],
            'sentiment_compound': vader_scores['compound']
        }

    def extract_emotional_load(self, text):
        """
        Quantify emotional content
        """
        # Tokenize text
        words = word_tokenize(text.lower())
        
        # Count emotion words
        emotion_counts = {
            emotion: sum(1 for word in words if word in emotion_words)
            for emotion, emotion_words in self.emotion_lexicon.items()
        }
        
        # Total emotional words ratio
        total_emotion_words = sum(emotion_counts.values())
        total_words = len(words)
        
        return {
            'emotional_density': total_emotion_words / total_words if total_words > 0 else 0,
            **emotion_counts
        }

    def extract_sentence_length(self, text):
        """
        Analyze sentence length characteristics
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Calculate sentence lengths
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        
        return {
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'max_sentence_length': max(sentence_lengths) if sentence_lengths else 0,
            'min_sentence_length': min(sentence_lengths) if sentence_lengths else 0,
            'sentence_length_std': np.std(sentence_lengths) if sentence_lengths else 0,
            'total_sentences': len(sentences)
        }

    def extract_all_features(self, text):
        """
        Combine all stylometric features
        """
        features = {}
        features.update(self.extract_syntactic_complexity(text))
        features.update(self.extract_sentiment(text))
        features.update(self.extract_emotional_load(text))
        features.update(self.extract_sentence_length(text))
        return features

def process_dataset(input_file, output_file):
    """
    Process entire dataset and extract stylometric features
    """
    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Initialize feature extractor
    extractor = StylometricFeatureExtractor()
    
    # Process each entry
    processed_data = []
    for entry in dataset:
        features = extractor.extract_all_features(entry['processed_text'])
        
        # Combine original entry with extracted features
        processed_entry = {
            'id': entry['id'],
            'processed_text': entry['processed_text'],
            'category': entry['category'],
            **features
        }
        processed_data.append(processed_entry)
    
    # Save processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    # Create DataFrame for further analysis
    df = pd.DataFrame(processed_data)
    return df

def main():
    # Define input and output paths
    base_path = "C:/Users/sebam/Desktop/NLP/progettoMereuNLP/pan-clef-2024-oppositional-main/pan-clef-2024-oppositional-main/dataset"
    
    # Input files
    train_input = os.path.join(base_path, 'dataset_en_train_processed.json')
    test_input = os.path.join(base_path, 'dataset_en_test_processed.json')
    
    # Output files
    train_output = os.path.join(base_path, 'stylometric_features_train.json')
    test_output = os.path.join(base_path, 'stylometric_features_test.json')
    
    # Process training dataset
    train_df = process_dataset(train_input, train_output)
    
    # Process test dataset
    test_df = process_dataset(test_input, test_output)
    
    # Comparative analysis
    print("\nStylometric Features - Training Set:")
    print(train_df.groupby('category').mean())
    
    print("\nStylometric Features - Test Set:")
    print(test_df.groupby('category').mean())
    
    # Advanced visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sentiment Comparison
    sentiment_cols = ['sentiment_positive', 'sentiment_negative', 'sentiment_compound']
    train_df.groupby('category')[sentiment_cols].mean().plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Sentiment Comparison')
    
    # Sentence Length Comparison
    length_cols = ['avg_sentence_length', 'max_sentence_length']
    train_df.groupby('category')[length_cols].mean().plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Sentence Length Comparison')
    
    # Emotional Density
    sns.boxplot(x='category', y='emotional_density', data=train_df, ax=axes[1, 0])
    axes[1, 0].set_title('Emotional Density Distribution')
    
    # Syntactic Complexity
    sns.boxplot(x='category', y='syntactic_complexity_score', data=train_df, ax=axes[1, 1])
    axes[1, 1].set_title('Syntactic Complexity Distribution')
    
    plt.tight_layout()
    plt.savefig('stylometric_features_analysis.png')
    plt.close()

if __name__ == "__main__":
    main()