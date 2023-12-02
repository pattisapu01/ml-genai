import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import os
import openai
from openai.types import CreateEmbeddingResponse, Embedding
import numpy as np
from scipy.spatial.distance import pdist, squareform
import concurrent.futures
from sklearn.decomposition import PCA
import datetime

# Load the dataset
df = pd.read_csv("simulated_transactions.csv")
now = datetime.datetime.now()
print("Current date and time: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))
# Numerical features
numerical_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']
features = df[numerical_columns].copy()

# Flag large transactions
large_transaction_threshold = 9000  # This threshold should be aligned with the one used in the data generation script
features['largeAmount'] = (df['amount'] > large_transaction_threshold).astype(int)

# Calculate balance changes for origin and destination
features['changebalanceOrig'] = features['newbalanceOrg'] - features['oldbalanceOrg']
features['changebalanceDest'] = features['newbalanceDest'] - features['oldbalanceDest']

# Extract hour from timestamp
# Ensure timestamp is a datetime object and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').sort_index()

# Count the number of transactions for each credit card within a certain time window
time_window = '1H'  # One hour time window
df['countTransac'] = df.groupby('creditCard').rolling(time_window)['amount'].count().reset_index(0, drop=True)

# Reset index after the rolling operation
df = df.reset_index()

# One-hot encoding for transaction type
type_one_hot = pd.get_dummies(df['type'])
features = pd.concat([features, type_one_hot], axis=1)


    
# Include the transaction count feature
features['countTransac'] = df['countTransac']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, df['isFraud'], test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle imbalanced dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Train the Random Forest model on the training set
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Evaluate the model on the test set
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

now = datetime.datetime.now()
print("Current date and time after random forest: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# embeddings

os.environ["OPENAI_API_KEY"] = "your key"
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to get embeddings from OpenAI
def get_embeddings_parallel(texts):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the executor to the function and texts
        futures = [executor.submit(get_single_embedding, text) for text in texts]
        return [future.result() for future in concurrent.futures.as_completed(futures)]

def get_single_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding




# Combine the test features with the corresponding part of the original dataframe to get all necessary columns
df_test = df.loc[X_test.index]
print("Number of rows in df_test:", len(df_test))

embeddings_file = 'embeddings_test.npy'


# Check if embeddings file exists
if os.path.exists(embeddings_file):
    print("Embeddings file FOUND!")
    # Load embeddings from file
    embeddings_array = np.load(embeddings_file)
else:
    print("Embeddings file NOT FOUND!")
    # Convert numerical fields to strings and concatenate them for embeddings
    df_test['combined'] = df_test['type'] + " " + df_test['amount'].astype(str) + " " + \
                      df_test['oldbalanceOrg'].astype(str) + " " + df_test['newbalanceOrg'].astype(str) + " " + \
                      df_test['oldbalanceDest'].astype(str) + " " + df_test['newbalanceDest'].astype(str) + " " + \
                      df_test['nameOrig'] + " " + df_test['nameDest'] + " " + df_test['timestamp'].astype(str) + " " + \
                      X_test['largeAmount'].astype(str) + " " + \
                      X_test['changebalanceOrig'].astype(str) + " " + \
                      X_test['changebalanceDest'].astype(str) + " " + \
                      X_test['countTransac'].astype(str)
    
    # For one-hot encoded features, add them as strings
    for col in type_one_hot.columns:
        if col in df_test.columns:
            df_test['combined'] += " " + col + "_" + df_test[col].astype(str)
        
    # Get embeddings for the combined column
    combined_texts = df_test['combined'].tolist()
    embeddings_list = get_embeddings_parallel(combined_texts)

    # Flatten each embedding to make it 1D and convert to a 2D array
    embeddings_array = np.array([np.array(embedding).flatten() for embedding in embeddings_list])

    # Save the embeddings to a file for future use
    np.save('embeddings_test.npy', embeddings_array)


# Normalize the embeddings using StandardScaler
scaler = StandardScaler()
try:
    normalized_embeddings = scaler.fit_transform(embeddings_array)
except ValueError as e:
    print("Error during normalization:", e)
    print("Shape of embeddings array before normalization:", embeddings_array.shape)


# Calculate cosine dissimilarity matrix (1 - cosine similarity)
cosine_dissimilarity_matrix = squareform(pdist(normalized_embeddings, 'cosine'))

# Find the threshold for anomalies
mean_dissimilarity = np.mean(cosine_dissimilarity_matrix)
std_dissimilarity = np.std(cosine_dissimilarity_matrix)
threshold = mean_dissimilarity + 1.5 * std_dissimilarity

# Identify indices of the anomalies
anomaly_indices = np.where(cosine_dissimilarity_matrix > threshold)

# Since cosine_dissimilarity_matrix is a square matrix, we get pairs of indices
anomaly_pairs = list(zip(anomaly_indices[0], anomaly_indices[1]))

# Now we need to map these pairs back to our transactions
anomaly_transactions = set()
for i, j in anomaly_pairs:
    if i != j:
        anomaly_transactions.add(i)
        anomaly_transactions.add(j)

# Create a mapping from new indices to original indices
index_mapping = df_test.index.tolist()

# Map the anomaly indices to original indices
mapped_anomaly_indices = [index_mapping[i] for i in anomaly_transactions]

# Add an 'embedding_cosine_isAnomaly' column to the DataFrame
df_test['embedding_cosine_isAnomaly'] = 0
df_test.loc[mapped_anomaly_indices, 'embedding_cosine_isAnomaly'] = 1

# Print the threshold value and anomaly count
print("Threshold for anomalies:", threshold)
print("Count of anomalies:", len(anomaly_transactions))
now = datetime.datetime.now()
print("Current date and time after embeddings: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))


# Calculate Euclidean distance matrix
euclidean_distance_matrix = squareform(pdist(normalized_embeddings, 'euclidean'))

# Find the threshold for anomalies using Euclidean distance
mean_euclidean = np.mean(euclidean_distance_matrix)
std_euclidean = np.std(euclidean_distance_matrix)
threshold_euclidean = mean_euclidean + 1.5 * std_euclidean

# Identify indices of the anomalies based on Euclidean distance
anomaly_indices_euclidean = np.where(euclidean_distance_matrix > threshold_euclidean)

# Function to extract anomalies from distance matrix
def extract_anomalies(distance_matrix, threshold):
    anomaly_indices = np.where(distance_matrix > threshold)
    anomaly_transactions = set()
    for i, j in zip(*anomaly_indices):
        if i != j:
            anomaly_transactions.add(i)
            anomaly_transactions.add(j)
    return anomaly_transactions

# Extract anomalies using Euclidean distance
euclidean_anomalies = extract_anomalies(euclidean_distance_matrix, threshold_euclidean)
# Create a mapping from new indices to original indices
index_mapping = df_test.index.tolist()

# Map the anomaly indices to original indices
mapped_euclidean_anomalies = [index_mapping[i] for i in euclidean_anomalies]

# Add an 'embedding_euclidean_isAnomaly' column to the DataFrame
df_test['embedding_euclidean_isAnomaly'] = 0
df_test.loc[mapped_euclidean_anomalies, 'embedding_euclidean_isAnomaly'] = 1

now = datetime.datetime.now()
print("Current date and time after euclidean distance: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))




# compare all approaches
# Function to calculate performance metrics


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

# Calculate metrics for embeddings approach on the test set
embedding_predictions = df_test.loc[X_test.index, 'embedding_cosine_isAnomaly']
embedding_accuracy, embedding_precision, embedding_recall, embedding_f1 = calculate_metrics(y_test, embedding_predictions)

# Calculate metrics for random forest approach on the test set
rf_predictions = rf_model.predict(X_test_scaled)
# Add the Random Forest predictions to df_test
df_test['rf_isAnomaly'] = rf_predictions
rf_accuracy, rf_precision, rf_recall, rf_f1 = calculate_metrics(y_test, rf_predictions)

# Save the DataFrame with the anomalies flagged
df_test.to_csv('transactions_with_anomalies.csv', index=False)

# Calculate metrics for Euclidean approach on the test set
euclidean_predictions = df_test.loc[X_test.index, 'embedding_euclidean_isAnomaly']
euclidean_accuracy, euclidean_precision, euclidean_recall, euclidean_f1 = calculate_metrics(y_test, euclidean_predictions)


# Compare anomalies detected by both methods on the test set
euclidean_anomalies = set(df_test.loc[X_test.index, 'embedding_euclidean_isAnomaly'][df_test.loc[X_test.index, 'embedding_euclidean_isAnomaly'] == 1].index)
embedding_anomalies = set(df_test.loc[X_test.index, 'embedding_cosine_isAnomaly'][df_test.loc[X_test.index, 'embedding_cosine_isAnomaly'] == 1].index)
rf_anomalies = set(X_test.index[rf_predictions == 1])
matched_anomalies = embedding_anomalies.intersection(rf_anomalies)
unique_to_embeddings = embedding_anomalies - rf_anomalies
unique_to_rf = rf_anomalies - embedding_anomalies
unique_to_euclidean = euclidean_anomalies - rf_anomalies - embedding_anomalies



# Generate a summary report
summary = {
    "Random Forest Approach": {
        "Accuracy": rf_accuracy,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1 Score": rf_f1,
        "Unique Anomalies": len(unique_to_rf)
    },
    "Embedding-Cosine Approach": {
        "Accuracy": embedding_accuracy,
        "Precision": embedding_precision,
        "Recall": embedding_recall,
        "F1 Score": embedding_f1,
        "Unique Anomalies": len(unique_to_embeddings)
    },
    "Embedding-Euclidean Approach": {
        "Accuracy": euclidean_accuracy,
        "Precision": euclidean_precision,
        "Recall": euclidean_recall,
        "F1 Score": euclidean_f1,
        "Unique Anomalies": len(unique_to_euclidean)
    },
    "Matched Anomalies": len(matched_anomalies)
}

# Print the summary report
for approach, metrics in summary.items():
    print(f"{approach}:")
    if isinstance(metrics, dict):
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    else:
        print(f"  {metrics}")
    print()


# Load the dataset
df_output = pd.read_csv('transactions_with_anomalies.csv')

# Define a function to calculate metrics
def calculate_metrics1(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Calculate metrics for each anomaly detection method
metrics_rf = calculate_metrics1(df_output['isFraud'], df_output['rf_isAnomaly'])
metrics_embedding = calculate_metrics1(df_output['isFraud'], df_output['embedding_cosine_isAnomaly'])
metrics_euclidean = calculate_metrics1(df_output['isFraud'], df_output['embedding_euclidean_isAnomaly'])

# Print the metrics
print("Random Forest Anomaly Detection Metrics:")
print("Accuracy:", metrics_rf[0], "Precision:", metrics_rf[1], "Recall:", metrics_rf[2], "F1 Score:", metrics_rf[3])

print("\nEmbedding-Cosine Anomaly Detection Metrics:")
print("Accuracy:", metrics_embedding[0], "Precision:", metrics_embedding[1], "Recall:", metrics_embedding[2], "F1 Score:", metrics_embedding[3])

print("\nEmbedding-Euclidean Anomaly Detection Metrics:")
print("Accuracy:", metrics_euclidean[0], "Precision:", metrics_euclidean[1], "Recall:", metrics_euclidean[2], "F1 Score:", metrics_euclidean[3])

# Set 'creditCard' as index
df_test.set_index('creditCard', inplace=True)
df_output.set_index('creditCard', inplace=True)

# Count matches between each anomaly detection method and isFraud
match_count_rf = (df_test['isFraud'] == df_output['rf_isAnomaly']).sum()
match_count_embedding = (df_test['isFraud'] == df_output['embedding_cosine_isAnomaly']).sum()
match_count_euclidean = (df_test['isFraud'] == df_output['embedding_euclidean_isAnomaly']).sum()

# Print match counts
total_fraud_count = df_test['isFraud'].sum()
print("Total count of isFraud = true rows [ground truth]:", total_fraud_count)
print("\nMatch Counts with isFraud:")
print("Random Forest Approach:", match_count_rf)
print("Embedding-Cosine Approach:", match_count_embedding)
print("Embedding-Euclidean Approach:", match_count_euclidean)

#neural net section

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load the dataset
df = pd.read_csv("simulated_transactions.csv")

# Preprocessing
# One-hot encoding for categorical features
encoder = OneHotEncoder(sparse=False)
type_encoded = encoder.fit_transform(df[['type']])

# Numerical features
numerical_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']
numerical_features = df[numerical_columns].copy()

# Scale the numerical features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(numerical_features)

# Combine numerical and categorical features
X = np.hstack((numerical_features_scaled, type_encoded))

# Target variable
y = df['isFraud'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Convert target variable to categorical (one-hot encoding)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Neural Network Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 output neurons for binary classification

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Summary of the model
model.summary()

# Training the model
epochs = 50  # Number of epochs for training
batch_size = 32  # Batch size for training

history = model.fit(X_train, y_train_categorical, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Evaluating the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Neural Network Model Accuracy: {accuracy}")

# Predictions
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Calculating metrics
from sklearn.metrics import precision_score, recall_score, f1_score

nn_precision = precision_score(y_test, y_pred)
nn_recall = recall_score(y_test, y_pred)
nn_f1 = f1_score(y_test, y_pred)

print(f"Neural Network Precision: {nn_precision}")
print(f"Neural Network Recall: {nn_recall}")
print(f"Neural Network F1 Score: {nn_f1}")

# Counting matched anomalies
nn_matched_anomalies = np.sum((y_pred == 1) & (y_test == 1))
print(f"Neural Network Matched Anomalies with Ground Truth: {nn_matched_anomalies} out of {total_fraud_count}")


now = datetime.datetime.now()
print("Current date and time after full run: ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))