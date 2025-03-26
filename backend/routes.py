from flask import Flask, request, jsonify
import pickle
import pandas as pd
from database import get_db_connection
from recommendation import recommend_properties

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, drop_first=True)
    
    prediction = model.predict(df)[0]
    return jsonify({'predicted_price': round(prediction, 2)})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    results = recommend_properties(data['location'], data['budget'])
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
