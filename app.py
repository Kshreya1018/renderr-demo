from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# --------------------------
# 1️⃣ Load the trained model
# --------------------------
model_path = 'best_model.pkl'  # Replace with your model filename
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# --------------------------
# 2️⃣ Initialize Flask app
# --------------------------
app = Flask(__name__)

# --------------------------
# 3️⃣ Define Home Route
# --------------------------
@app.route('/')
def home():
    # Renders your HTML frontend (optional)
    return render_template('index.html')

# --------------------------
# 4️⃣ Define Prediction Route
# --------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form (HTML) or API request (Flutter/Postman)
        data = request.form.get('features')  # for form data
        # OR for JSON data from mobile app:
        # data = request.get_json()['features']

        # Convert input to array
        # Example: input "1,2,3,4" → [1.0, 2.0, 3.0, 4.0]
        features = np.array([float(x) for x in data.split(',')]).reshape(1, -1)

        # Predict using the ML model
        prediction = model.predict(features)

        # Convert output to readable form
        output = prediction[0]

        return jsonify({'prediction': str(output)})

    except Exception as e:
        return jsonify({'error': str(e)})

# --------------------------
# 5️⃣ Run the Flask server
# --------------------------
if __name__ == '_main_':
    app.run(debug=True)