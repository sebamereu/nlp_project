import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import argparse
import numpy as np
import spacy
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

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
        """Extract syntactic complexity features"""
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
        """Comprehensive sentiment analysis"""
        # VADER sentiment scores
        vader_scores = self.sia.polarity_scores(text)
        
        return {
            'sentiment_positive': vader_scores['pos'],
            'sentiment_negative': vader_scores['neg'],
            'sentiment_neutral': vader_scores['neu'],
            'sentiment_compound': vader_scores['compound']
        }

    def extract_emotional_load(self, text):
        """Quantify emotional content"""
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
        """Analyze sentence length characteristics"""
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
        """Combine all stylometric features"""
        features = {}
        features.update(self.extract_syntactic_complexity(text))
        features.update(self.extract_sentiment(text))
        features.update(self.extract_emotional_load(text))
        features.update(self.extract_sentence_length(text))
        return features

class DatasetAnalyzer:
    def __init__(self, file_path, base_plot_dir='plot'):
        # Validate the file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        # Load the dataset
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)
        except json.JSONDecodeError:
            raise ValueError(f"The file {file_path} is not a valid JSON.")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        self.validate_columns(['category', 'text'])  # Check for required columns
        
        # Extract dataset name for titles and directories
        self.dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create a specific directory for plots based on the dataset name
        self.plot_dir = os.path.join(base_plot_dir, self.dataset_name)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Initialize stylometric feature extractor
        self.stylometric_extractor = StylometricFeatureExtractor()

    def validate_columns(self, required_columns):
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"The dataset is missing the required columns: {', '.join(missing_columns)}")

    def analyze_class_distribution(self):
        # (Previous implementation remains the same)
        class_distribution = self.df['category'].value_counts()
        class_percentages = class_distribution / len(self.df) * 100

        self.stats_df = pd.DataFrame({
            'Class': class_distribution.index,
            'Count': class_distribution.values,
            'Percentage': class_percentages.values
        })

        print("\n1. Class Distribution:")
        print(self.stats_df)
        print(f"\nDataset Size: {len(self.df)}")
        print(f"Class Counts:\n{class_distribution}")

        # Save statistics
        self.stats_df.to_csv(os.path.join(self.plot_dir, 'class_distribution.csv'), index=False)

        # Plot class distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='Count', hue='Class', data=self.stats_df, palette='viridis', dodge=False)
        plt.title(f'Class Distribution in the {self.dataset_name}')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'class_distribution.png'), dpi=300)
        plt.close()

    def analyze_text_characteristics(self):
        # Add text length and word count
        self.df['text_length'] = self.df['text'].str.len()
        self.df['word_count'] = self.df['text'].apply(lambda x: len(word_tokenize(x)))

        length_stats = self.df.groupby('category')['text_length'].agg(['mean', 'median', 'min', 'max', 'std'])
        word_stats = self.df.groupby('category')['word_count'].agg(['mean', 'median', 'min', 'max', 'std'])

        print("\n2. Text Length Statistics:")
        print(length_stats)
        print("\n3. Word Count Statistics:")
        print(word_stats)

        # Save statistics
        length_stats.to_csv(os.path.join(self.plot_dir, 'length_stats.csv'))
        word_stats.to_csv(os.path.join(self.plot_dir, 'word_stats.csv'))

        # Boxplot graphs
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df, x='category', y='text_length', palette='Set3')
        plt.title(f'Text Length Distribution in the {self.dataset_name}')
        plt.xlabel('Classes')
        plt.ylabel('Text Length (characters)')

        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x='category', y='word_count', palette='Set2')
        plt.title(f'Word Count Distribution in the {self.dataset_name}')
        plt.xlabel('Classes')
        plt.ylabel('Number of Words')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'text_characteristics.png'), dpi=300)
        plt.close()

    def analyze_vocabulary_overview(self):
        def count_unique_words(texts):
            all_words = ' '.join(texts).split()
            return len(set(all_words))

        vocab_stats = self.df.groupby('category')['text'].agg([
            ('total_texts', 'count'),
            ('unique_words', lambda x: count_unique_words(x))
        ])

        print("\n4. Vocabulary Overview:")
        print(vocab_stats)

        # Save statistics
        vocab_stats.to_csv(os.path.join(self.plot_dir, 'vocabulary_stats.csv'))

    def analyze_stylometric_features(self):
        # Extract stylometric features for each text
        #CORE della funzione
        stylometric_features = self.df['text'].apply(self.stylometric_extractor.extract_all_features)
        
        # Convert features to DataFrame
        stylometric_df = pd.DataFrame(stylometric_features.tolist())
        
        # Combine with original DataFrame
        self.df = pd.concat([self.df, stylometric_df], axis=1)
        
        # Aggregate stylometric features by category
        feature_columns = [
            'sentiment_positive', 'sentiment_negative', 'sentiment_compound', 
            'emotional_density', 'avg_sentence_length', 
            'syntactic_complexity_score'
        ]
        
        stylometric_stats = self.df.groupby('category')[feature_columns].agg(['mean', 'std'])
        
        print("\n5. Stylometric Features Overview:")
        print(stylometric_stats)
        
        # Save stylometric statistics
        stylometric_stats.to_csv(os.path.join(self.plot_dir, 'stylometric_stats.csv'))
        
        # Visualize stylometric features
        plt.figure(figsize=(15, 10))
        
        # Sentiment Features
        plt.subplot(2, 2, 1)
        sns.barplot(x=stylometric_stats.index, 
                    y=stylometric_stats[('sentiment_compound', 'mean')], 
                    palette='viridis')
        plt.title('Mean Sentiment Compound Score')
        plt.xticks(rotation=45)
        
        # Emotional Density
        plt.subplot(2, 2, 2)
        sns.boxplot(x='category', y='emotional_density', data=self.df, palette='Set2')
        plt.title('Emotional Density Distribution')
        plt.xticks(rotation=45)
        
        # Sentence Length
        plt.subplot(2, 2, 3)
        sns.boxplot(x='category', y='avg_sentence_length', data=self.df, palette='Set3')
        plt.title('Average Sentence Length')
        plt.xticks(rotation=45)
        
        # Syntactic Complexity
        plt.subplot(2, 2, 4)
        sns.boxplot(x='category', y='syntactic_complexity_score', data=self.df, palette='Set1')
        plt.title('Syntactic Complexity Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'stylometric_features.png'), dpi=300)
        plt.close()

    def full_analysis(self, 
                      perform_distribution=True, 
                      perform_text_analysis=True, 
                      perform_vocabulary=True, 
                      perform_stylometric=True):
        if perform_distribution:
            self.analyze_class_distribution()
        if perform_text_analysis:
            self.analyze_text_characteristics()
        if perform_vocabulary:
            self.analyze_vocabulary_overview()
        if perform_stylometric:
            self.analyze_stylometric_features()


# Argparse for paths and options
def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Analyzer")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset JSON file')
    parser.add_argument('--plot_dir', type=str, default='plot', help='Base directory to save plots')
    parser.add_argument('--skip_distribution', action='store_true', help='Skip class distribution analysis')
    parser.add_argument('--skip_text_analysis', action='store_true', help='Skip text characteristics analysis')
    parser.add_argument('--skip_vocabulary', action='store_true', help='Skip vocabulary overview analysis')
    parser.add_argument('--skip_stylometric', action='store_true', help='Skip stylometric feature analysis')
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    analyzer = DatasetAnalyzer(args.file_path, args.plot_dir)
    analyzer.full_analysis(
        perform_distribution=not args.skip_distribution,
        perform_text_analysis=args.skip_text_analysis,
        perform_vocabulary=args.skip_vocabulary,
        perform_stylometric=args.skip_stylometric
    )