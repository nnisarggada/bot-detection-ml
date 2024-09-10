# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import sys

app = Flask(__name__)

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
    model, scaler = load_model_and_scaler()
    print("Loaded pre-trained model and scaler.")
else:
    train_and_save_model()
    model, scaler = load_model_and_scaler()

# Function to predict the visitor type
def predict(features):
    # Create DataFrame with column names
    features_df = pd.DataFrame([features], columns=['Page Views', 'Session Duration', 'Time on Page', 'Previous Visits'])
    # Scale the features
    scaled_features = scaler.transform(features_df)
    # Predict
    prediction = model.predict(scaled_features)
    return 'Bot' if prediction[0] == 1 else 'Human'

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Get the JSON data from the POST request
        data = request.get_json()

        # Extract the necessary features from the data
        page_views = data.get('page_views', None)
        session_duration = data.get('session_duration', None)
        time_on_page = data.get('time_on_page', None)
        previous_visits = data.get('previous_visits', None)

        # Ensure all features are provided
        if None in (page_views, session_duration, time_on_page, previous_visits):
            return jsonify({'error': 'Missing one or more features.'}), 400

        # Format the input for prediction
        features = [float(page_views), float(session_duration), float(time_on_page), int(previous_visits)]

        # Get the prediction
        result = predict(features)

        # Return the prediction as JSON
        return jsonify({'prediction': result}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1].split('=')[-1])
    
    app.run(port=port, debug=True)
