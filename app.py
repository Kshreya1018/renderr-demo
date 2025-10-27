from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os 
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'best_model (1).pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from: {model_path}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"The model file was not found at {model_path}. "
        "Please ensure 'best_model.pkl' is in the same directory as this script."
    )
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {e}")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.get('features') 
    
        features = np.array([float(x) for x in data.split(',')]).reshape(1, -1)

        prediction = model.predict(features)

        output = prediction[0]

        return jsonify({'prediction': str(output)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
   
    app.run(debug=True)