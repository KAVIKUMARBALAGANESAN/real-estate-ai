from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from database import get_db_connection
from recommendation import recommend_properties

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load trained ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return jsonify({"message": "Real Estate Price Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Convert categorical variables (ensure it matches the model training)
        df = pd.get_dummies(df, drop_first=True)

        # Predict the property price
        prediction = model.predict(df)[0]
        return jsonify({'predicted_price': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        location = data.get('location')
        budget = data.get('budget')

        recommendations = recommend_properties(location, budget)
        return jsonify(recommendations)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
