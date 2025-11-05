from flask import Flask, render_template, request
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open("wine_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get all input values from form
    values = [float(x) for x in request.form.values()]

    # Convert to numpy array and reshape
    final_input = np.array(values).reshape(1, -1)

    # Scale input
    final_input = scaler.transform(final_input)

    # Make prediction
    output = model.predict(final_input)[0]

    result = "Good Quality Wine 🍷" if output == 1 else "Bad Quality Wine ❌"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
