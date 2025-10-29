from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained models
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = heart_model.predict(final_features)[0]
    
    result = "💓 Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease"
    return render_template('index.html', prediction_text=result)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    prediction = diabetes_model.predict(final_features)[0]
    
    result = "🍬 Diabetes Detected" if prediction == 1 else "✅ No Diabetes"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
