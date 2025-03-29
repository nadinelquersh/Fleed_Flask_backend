from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model from SavedModel format
model = tf.keras.models.load_model("saved_model/")

@app.route("/")
def home():
    return "AI Model Backend is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["input"]
        input_tensor = np.array(data).reshape(1, 128, 128, 3)  # Adjust shape as needed
        prediction = model.predict(input_tensor).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
