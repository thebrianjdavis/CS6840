# Import necessary modules
import pandas as pd
import torch.backends.mps
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocessing import preprocess_data
from feature_extraction import extract_features
import logging

# Set up the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
df = pd.read_csv('Datasets/mini_lyrics_raw.csv', on_bad_lines='skip')

# Preprocess data
preprocessed_df = preprocess_data(df)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_df['lyrics'], preprocessed_df['tag'], test_size=0.2, random_state=42)

# Extract features
logging.info("Starting feature extraction...")
X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)
logging.info("Feature extraction completed.")

# Train SVM classifier with class weights
class_weights = 'balanced'  # This automatically adjusts weights inversely proportional to class frequencies


logging.info("Starting model training...")
svm = SVC(kernel='linear', class_weight='balanced')
svm.fit(X_train_features, y_train)
logging.info("Model training completed.")

# Predict on the test set
y_pred = svm.predict(X_test_features)

# Evaluate the model
print(classification_report(y_test, y_pred))
