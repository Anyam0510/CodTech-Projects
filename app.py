from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load("iris_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is up!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)