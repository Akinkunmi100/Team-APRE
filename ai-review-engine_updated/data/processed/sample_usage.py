
# Load the processed data
import pandas as pd
from utils.data_preprocessing import DataPreprocessor
from models.absa_model import ABSASentimentAnalyzer

# Load data
df = pd.read_csv('data/processed/reviews_combined.csv')

# Initialize analyzer
analyzer = ABSASentimentAnalyzer()
preprocessor = DataPreprocessor()

# Preprocess and analyze
df = preprocessor.preprocess_dataset(df)
results = analyzer.analyze_batch(df.to_dict('records'))

# View results
print(f"Total reviews analyzed: {len(results)}")
print(f"Average sentiment: {results['sentiment'].value_counts()}")
