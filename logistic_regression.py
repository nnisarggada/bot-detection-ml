import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# Path to the model file
model_filename = 'pretrained/model.pkl'
scaler_filename = 'pretrained/scaler.pkl'

# Load data from CSV file
df = pd.read_csv('data/balanced_data.csv')

# Features and target variable
x = df[['Page Views', 'Session Duration', 'Time on Page', 'Previous Visits']]
y = df['Visitor Type'].apply(lambda x: 1 if x == 'Bot' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Function to train and save the model and scaler
def train_and_save_model():
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Save the model and scaler
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print("Model and scaler trained and saved.")

# Function to load the model and scaler
def load_model_and_scaler():
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Check if the pre-trained model exists
if os.path.exists(model_filename) and os.path.exists(scaler_filename):
    user_choice = input("A pre-trained model exists. Do you want to use the pre-trained model? (Y/n): ").strip().lower()
    if user_choice == 'n' or user_choice == 'no':
        train_and_save_model()
        model, scaler = load_model_and_scaler()
    else:
        model, scaler = load_model_and_scaler()
        print("Loaded pre-trained model and scaler.")
else:
    train_and_save_model()
    model, scaler = load_model_and_scaler()

# Predict on test data
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

# Function to make predictions based on user input
def predict(visitor_features):
    # Create DataFrame with column names
    features_df = pd.DataFrame([visitor_features], columns=['Page Views', 'Session Duration', 'Time on Page', 'Previous Visits'])
    # Scale the features
    scaled_features = scaler.transform(features_df)
    # Predict
    prediction = model.predict(scaled_features)
    return 'Bot' if prediction[0] == 1 else 'Human'

# Input from user
try:
    page_views = float(input("Enter Page Views: "))
    session_duration = float(input("Enter Session Duration (seconds): "))
    time_on_page = float(input("Enter Time on Page (seconds): "))
    previous_visits = int(input("Enter Previous Visits: "))

    features = [page_views, session_duration, time_on_page, previous_visits]
    result = predict(features)
    print(f"The visitor is predicted to be: {result}")
except ValueError:
    print("Invalid input. Please enter numeric values for the features.")
