import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import argparse

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize

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

    def validate_columns(self, required_columns):
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"The dataset is missing the required columns: {', '.join(missing_columns)}")

    def analyze_class_distribution(self):
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

    def full_analysis(self, perform_distribution=True, perform_text_analysis=True, perform_vocabulary=True):
        if perform_distribution:
            self.analyze_class_distribution()
        if perform_text_analysis:
            self.analyze_text_characteristics()
        if perform_vocabulary:
            self.analyze_vocabulary_overview()


# Argparse for paths and options
def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Analyzer")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the dataset JSON file')
    parser.add_argument('--plot_dir', type=str, default='plot', help='Base directory to save plots')
    parser.add_argument('--skip_distribution', action='store_true', help='Skip class distribution analysis')
    parser.add_argument('--skip_text_analysis', action='store_true', help='Skip text characteristics analysis')
    parser.add_argument('--skip_vocabulary', action='store_true', help='Skip vocabulary overview analysis')
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    analyzer = DatasetAnalyzer(args.file_path, args.plot_dir)
    analyzer.full_analysis(
        perform_distribution=not args.skip_distribution,
        perform_text_analysis=not args.skip_text_analysis,
        perform_vocabulary=not args.skip_vocabulary
    )