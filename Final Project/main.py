# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from preprocessing import preprocess_data
from feature_extraction import extract_features_sbert, extract_features_roberta
import logging
import time

# Select dataset
# dataset = './Datasets/mini_lyrics_raw.csv'
# dataset = './Datasets/lyrics_raw.csv'
dataset = './Datasets/balanced_dataset_small.csv'

# Set up the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data
df = pd.read_csv(dataset, on_bad_lines='skip')

# Preprocess data
preprocessed_df = preprocess_data(df)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    preprocessed_df['lyrics'], preprocessed_df['tag'], test_size=0.2, random_state=42)

# Extract features using SBERT model
extraction_start_sbert = time.time()
logging.info("Starting SBERT feature extraction...")
X_train_features_sbert = extract_features_sbert(X_train)
X_test_features_sbert = extract_features_sbert(X_test)
logging.info("SBERT feature extraction completed.")
extraction_end_sbert = time.time()

# Extract features using RoBERTa model
extraction_start_roberta = time.time()
logging.info("Starting RoBERTa feature extraction...")
X_train_features_roberta = extract_features_roberta(X_train)
X_test_features_roberta = extract_features_roberta(X_test)
logging.info("RoBERTa feature extraction completed.")
extraction_end_roberta = time.time()

# Train SVM classifier with class weights using SBERT model
training_start_sbert = time.time()
logging.info("Starting SBERT model training...")
svm_sbert = SVC(kernel='linear', class_weight='balanced')
svm_sbert.fit(X_train_features_sbert, y_train)
logging.info("SBERT model training completed.")
training_end_sbert = time.time()

# Train SVM classifier with class weights using RoBERTa model
training_start_roberta = time.time()
logging.info("Starting RoBERTa model training...")
svm_roberta = SVC(kernel='linear', class_weight='balanced')
svm_roberta.fit(X_train_features_roberta, y_train)
logging.info("RoBERTa model training completed.")
training_end_roberta = time.time()

# Predict on the test set using SBERT model
prediction_start_sbert = time.time()
logging.info("Starting test set predictions...")
y_pred_sbert = svm_sbert.predict(X_test_features_sbert)
logging.info("Test set predictions completed.")
prediction_end_sbert = time.time()

# Predict on the test set using RoBERTa model
prediction_start_roberta = time.time()
logging.info("Starting test set predictions...")
y_pred_roberta = svm_roberta.predict(X_test_features_roberta)
logging.info("Test set predictions completed.")
prediction_end_roberta = time.time()

# Open a file to write the results
with open('model_results.txt', 'w') as file:
    # Evaluate the models and print to file
    print("############ SBERT Classification Report ############", file=file)
    print(classification_report(y_test, y_pred_sbert), file=file)
    print("############ RoBERTa Classification Report ############", file=file)
    print(classification_report(y_test, y_pred_roberta), file=file)

    # Calculate processing times for SBERT model and print to file
    extraction_time_sbert = extraction_end_sbert - extraction_start_sbert
    training_time_sbert = training_end_sbert - training_start_sbert
    prediction_time_sbert = prediction_end_sbert - prediction_start_sbert
    print("############ SBERT Processing Time ############", file=file)
    print("Feature Extraction Processing time:", extraction_time_sbert, "seconds", file=file)
    print("Model Training Processing time:", training_time_sbert, "seconds", file=file)
    print("Prediction Processing time:", prediction_time_sbert, "seconds", file=file)

    # Calculate processing times for RoBERTa model and print to file
    extraction_time_roberta = extraction_end_roberta - extraction_start_roberta
    training_time_roberta = training_end_roberta - training_start_roberta
    prediction_time_roberta = prediction_end_roberta - prediction_start_roberta
    print("\n############ RoBERTa Processing Time ############", file=file)
    print("Feature Extraction Processing time:", extraction_time_roberta, "seconds", file=file)
    print("Model Training Processing time:", training_time_roberta, "seconds", file=file)
    print("Prediction Processing time:", prediction_time_roberta, "seconds", file=file)
