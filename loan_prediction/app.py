from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained pipeline (preprocess + model)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_pipeline.pkl")
clf = joblib.load(MODEL_PATH)

# The raw feature names the pipeline expects (same as training)
FEATURES = [
    'Gender',
    'Married',
    'Dependents',
    'Education',
    'Self_Employed',
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History',
    'Property_Area'
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect raw values from the form (as strings), convert to proper types
        form = request.form

        # Note: we pass the SAME raw column names as during training; the pipeline will
        # impute + one-hot encode internally, so we avoid feature-count mismatches.
        row = {
            'Gender': form.get('Gender'),                     # "Male"/"Female"
            'Married': form.get('Married'),                   # "Yes"/"No"
            'Dependents': form.get('Dependents'),             # "0","1","2","3+"
            'Education': form.get('Education'),               # "Graduate"/"Not Graduate"
            'Self_Employed': form.get('Self_Employed'),       # "Yes"/"No"
            'ApplicantIncome': float(form.get('ApplicantIncome', 0) or 0),
            'CoapplicantIncome': float(form.get('CoapplicantIncome', 0) or 0),
            'LoanAmount': float(form.get('LoanAmount', 0) or 0),
            'Loan_Amount_Term': float(form.get('Loan_Amount_Term', 0) or 0),
            'Credit_History': float(form.get('Credit_History', 1) or 1),
            'Property_Area': form.get('Property_Area')        # "Urban"/"Semiurban"/"Rural"
        }

        # Convert "3+" dependents to "3" or keep as "3+" consistently with training data
        # In training, "Dependents" was categorical -> OHE; either "3+" is okay.
        # If your CSV uses "3+" keep it; else normalize:
        if row['Dependents'] == '3':
            row['Dependents'] = '3'  # ok
        elif row['Dependents'] == '3+':
            row['Dependents'] = '3+' # ok
        # (If your CSV has numerics for Dependents, replace with numeric string: '3')

        # Build a single-row DataFrame with the exact raw column names
        input_df = pd.DataFrame([row], columns=FEATURES)

        # Predict
        pred = clf.predict(input_df)[0]
        proba = None
        try:
            proba = float(clf.predict_proba(input_df)[0][1])
        except Exception:
            pass

        if pred == 1:
            msg = "Loan Approved"
        else:
            msg = "Loan Rejected"

        if proba is not None:
            msg = f"{msg} (Probability: {proba:.2%})"

        return render_template("index.html", prediction=msg, last_input=row)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
