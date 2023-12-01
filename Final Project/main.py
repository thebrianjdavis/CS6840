# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from preprocessing import preprocess_data
from feature_extraction import extract_features_sbert, extract_features_roberta
import logging
import time

process_start = time.time()

# Select dataset
# dataset = './Datasets/mini_lyrics_raw.csv'
# dataset = './Datasets/lyrics_raw.csv'
# dataset = './Datasets/balanced_dataset_small.csv'
# dataset = './Datasets/balanced_dataset_medium.csv'
dataset = './Datasets/balanced_dataset_large.csv'

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
training_start_sbert_svm = time.time()
logging.info("Starting SBERT model training...")
sbert_svm = SVC(kernel='linear', class_weight='balanced')
sbert_svm.fit(X_train_features_sbert, y_train)
logging.info("SBERT model training completed.")
training_end_sbert_svm = time.time()

# Train SVM classifier with class weights using RoBERTa model
training_start_roberta_svm = time.time()
logging.info("Starting RoBERTa model training...")
roberta_svm = SVC(kernel='linear', class_weight='balanced')
roberta_svm.fit(X_train_features_roberta, y_train)
logging.info("RoBERTa model training completed.")
training_end_roberta_svm = time.time()

# Train Logistic Regression classifier using SBERT model
training_start_sbert_lr = time.time()
logging.info("Starting Logistic Regression training with SBERT features...")
sbert_log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
sbert_log_reg.fit(X_train_features_sbert, y_train)
logging.info("Logistic Regression training with SBERT features completed.")
training_end_sbert_lr = time.time()

# Train Logistic Regression classifier using SBERT model
training_start_roberta_lr = time.time()
logging.info("Starting Logistic Regression training with RoBERTa features...")
roberta_log_reg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
roberta_log_reg.fit(X_train_features_roberta, y_train)
logging.info("Logistic Regression training with RoBERTa features completed.")
training_end_roberta_lr = time.time()

# Train Random Forest classifier using SBERT model
training_start_sbert_rf = time.time()
logging.info("Starting Random Forest training with SBERT features...")
sbert_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
sbert_rf.fit(X_train_features_sbert, y_train)
logging.info("Random Forest training with SBERT features completed.")
training_end_sbert_rf = time.time()

# Train Random Forest classifier using RoBERTa model
training_start_roberta_rf = time.time()
logging.info("Starting Random Forest training with RoBERTa features...")
roberta_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
roberta_rf.fit(X_train_features_roberta, y_train)
logging.info("Random Forest training with RoBERTa features completed.")
training_end_roberta_rf = time.time()

# Train MLP classifier using SBERT model
training_start_sbert_mlp = time.time()
logging.info("Starting MLP training with SBERT features...")
sbert_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, activation='relu', solver='adam', random_state=42)
sbert_mlp.fit(X_train_features_sbert, y_train)
logging.info("MLP training with SBERT features completed.")
training_end_sbert_mlp = time.time()

# Train MLP classifier using RoBERTa model
training_start_roberta_mlp = time.time()
logging.info("Starting MLP training with RoBERTa features...")
roberta_mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, activation='relu', solver='adam', random_state=42)
roberta_mlp.fit(X_train_features_roberta, y_train)
logging.info("MLP training with RoBERTa features completed.")
training_end_roberta_mlp = time.time()

# Predict on the test set using SBERT/SVM model
prediction_start_sbert_svm = time.time()
logging.info("Starting test set SBERT/SVM predictions...")
y_pred_sbert_svm = sbert_svm.predict(X_test_features_sbert)
logging.info("SBERT/SVM test set predictions completed.")
prediction_end_sbert_svm = time.time()

# Predict on the test set using RoBERTa/SVM model
prediction_start_roberta_svm = time.time()
logging.info("Starting test set RoBERTa/SVM predictions...")
y_pred_roberta_svm = roberta_svm.predict(X_test_features_roberta)
logging.info("RoBERTa/SVM test set predictions completed.")
prediction_end_roberta_svm = time.time()

# Predict on the test set using SBERT/LR model
prediction_start_sbert_lr = time.time()
logging.info("Starting test set SBERT/LR predictions...")
y_pred_sbert_lr = sbert_log_reg.predict(X_test_features_sbert)
logging.info("SBERT/LR test set predictions completed.")
prediction_end_sbert_lr = time.time()

# Predict on the test set using RoBERTa/LR model
prediction_start_roberta_lr = time.time()
logging.info("Starting test set RoBERTa/LR predictions...")
y_pred_roberta_lr = roberta_log_reg.predict(X_test_features_roberta)
logging.info("RoBERTa/LR est set predictions completed.")
prediction_end_roberta_lr = time.time()

# Predict on the test set using SBERT/Random Forest model
prediction_start_sbert_rf = time.time()
logging.info("Starting test set SBERT/Random Forest predictions...")
y_pred_sbert_rf = sbert_rf.predict(X_test_features_sbert)
logging.info("SBERT/Random Forest test set predictions completed.")
prediction_end_sbert_rf = time.time()

# Predict on the test set using RoBERTa/Random Forest model
prediction_start_roberta_rf = time.time()
logging.info("Starting test set RoBERTa/Random Forest predictions...")
y_pred_roberta_rf = roberta_rf.predict(X_test_features_roberta)
logging.info("RoBERTa/Random Forest test set predictions completed.")
prediction_end_roberta_rf = time.time()

# Predict on the test set using SBERT/MLP model
prediction_start_sbert_mlp = time.time()
logging.info("Starting test set SBERT/MLP predictions...")
y_pred_sbert_mlp = sbert_mlp.predict(X_test_features_sbert)
logging.info("SBERT/MLP test set predictions completed.")
prediction_end_sbert_mlp = time.time()

# Predict on the test set using RoBERTa/MLP model
prediction_start_roberta_mlp = time.time()
logging.info("Starting test set RoBERTa/MLP predictions...")
y_pred_roberta_mlp = roberta_mlp.predict(X_test_features_roberta)
logging.info("RoBERTa/MLP test set predictions completed.")
prediction_end_roberta_mlp = time.time()

process_end = time.time()

# Open a file to write the results
with open('model_results.txt', 'w') as file:
    print("################ QUALITATIVE ANALYSIS ###############", file=file)
    # Evaluate the models and print to file
    print("########## SBERT/SVM Classification Report ##########", file=file)
    print(classification_report(y_test, y_pred_sbert_svm, zero_division=1), file=file)
    print("######### RoBERTa/SVM Classification Report #########", file=file)
    print(classification_report(y_test, y_pred_roberta_svm, zero_division=1), file=file)
    print("########## SBERT/LR Classification Report ###########", file=file)
    print(classification_report(y_test, y_pred_sbert_lr, zero_division=1), file=file)
    print("######### RoBERTa/LR Classification Report ##########", file=file)
    print(classification_report(y_test, y_pred_roberta_lr, zero_division=1), file=file)
    print("##### SBERT/Random Forest Classification Report #####", file=file)
    print(classification_report(y_test, y_pred_sbert_rf, zero_division=1), file=file)
    print("#### RoBERTa/Random Forest Classification Report ####", file=file)
    print(classification_report(y_test, y_pred_roberta_rf, zero_division=1), file=file)
    print("########## SBERT/MLP Classification Report ##########", file=file)
    print(classification_report(y_test, y_pred_sbert_mlp, zero_division=1), file=file)
    print("######### RoBERTa/MLP Classification Report #########", file=file)
    print(classification_report(y_test, y_pred_roberta_mlp, zero_division=1), file=file)

    print("\n################ QUANTITATIVE ANALYSIS ##############", file=file)

    # Calculate processing times for feature extraction and print to file
    extraction_time_sbert = extraction_end_sbert - extraction_start_sbert
    extraction_time_roberta = extraction_end_roberta - extraction_start_roberta
    print("\n######## Feature Extraction Processing Time #########", file=file)
    print("SBERT:", extraction_time_sbert, "seconds", file=file)
    print("RoBERTa:", extraction_time_roberta, "seconds", file=file)

    # Calculate processing times for model training and print to file
    training_time_sbert_svm = training_end_sbert_svm - training_start_sbert_svm
    training_time_roberta_svm = training_end_roberta_svm - training_start_roberta_svm
    training_time_sbert_lr = training_end_sbert_lr - training_start_sbert_lr
    training_time_roberta_lr = training_end_roberta_lr - training_start_roberta_lr
    training_time_sbert_rf = training_end_sbert_rf - training_start_sbert_rf
    training_time_roberta_rf = training_end_roberta_rf - training_start_roberta_rf
    training_time_sbert_mlp = training_end_sbert_mlp - training_start_sbert_mlp
    training_time_roberta_mlp = training_end_roberta_mlp - training_start_roberta_mlp
    print("\n########## Model Training Processing Time ###########", file=file)
    print("SBERT/SVM:", training_time_sbert_svm, "seconds", file=file)
    print("RoBERTa/SVM:", training_time_roberta_svm, "seconds", file=file)
    print("SBERT/LR:", training_time_sbert_lr, "seconds", file=file)
    print("RoBERTa/LR:", training_time_roberta_lr, "seconds", file=file)
    print("SBERT/Random Forest:", training_time_sbert_rf, "seconds", file=file)
    print("RoBERTa/Random Forest:", training_time_roberta_rf, "seconds", file=file)
    print("SBERT/MLP:", training_time_sbert_mlp, "seconds", file=file)
    print("RoBERTa/MLP:", training_time_roberta_mlp, "seconds", file=file)

    # Calculate processing times for predictions and print to file
    prediction_time_sbert_svm = prediction_end_sbert_svm - prediction_start_sbert_svm
    prediction_time_roberta_svm = prediction_end_roberta_svm - prediction_start_roberta_svm
    prediction_time_sbert_lr = prediction_end_sbert_lr - prediction_start_sbert_lr
    prediction_time_roberta_lr = prediction_end_roberta_lr - prediction_start_roberta_lr
    prediction_time_sbert_rf = prediction_end_sbert_rf - prediction_start_sbert_rf
    prediction_time_roberta_rf = prediction_end_roberta_rf - prediction_start_roberta_rf
    prediction_time_sbert_mlp = prediction_end_sbert_mlp - prediction_start_sbert_mlp
    prediction_time_roberta_mlp = prediction_end_roberta_mlp - prediction_start_roberta_mlp
    print("\n############ Prediction Processing Time #############", file=file)
    print("SBERT/SVM:", prediction_time_sbert_svm, "seconds", file=file)
    print("RoBERTa/SVM:", prediction_time_roberta_svm, "seconds", file=file)
    print("SBERT/LR:", prediction_time_sbert_lr, "seconds", file=file)
    print("RoBERTa/LR:", prediction_time_roberta_lr, "seconds", file=file)
    print("SBERT/Random Forest:", prediction_time_sbert_rf, "seconds", file=file)
    print("RoBERTa/Random Forest:", prediction_time_roberta_rf, "seconds", file=file)
    print("SBERT/MLP:", prediction_time_sbert_mlp, "seconds", file=file)
    print("RoBERTa/MLP:", prediction_time_roberta_mlp, "seconds", file=file)

    # Calculate total run time and print to file
    process_time = process_end - process_start
    print("\n################## Total Run Time ###################", file=file)
    minutes = process_time // 60
    seconds = process_time - (minutes * 60)
    print("TOTAL RUN TIME:", minutes, "minutes", seconds, "seconds", file=file)
