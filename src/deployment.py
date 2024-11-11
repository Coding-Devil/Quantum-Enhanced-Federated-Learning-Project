# src/deployment.py

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('path/to/your/trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Preprocess data as needed
    # prediction = model.predict(data)  # Adjust as necessary
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)