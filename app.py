from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()  # loads .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def train_model(data, target_col, feature_cols):
    """Trains a Logistic Regression model and returns model, scaler, features."""
    available_cols = [c for c in feature_cols if c in data.columns]
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found! Found: {data.columns.tolist()}")

    df = data[available_cols + [target_col]].dropna().copy()

    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df[available_cols]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler, available_cols

diabetes = pd.read_csv("diabetes.csv")
diabetes_features = ['Pregnancies', 'Glucose', 'BMI', 'Age']
model_diabetes, scaler_d, diabetes_features = train_model(diabetes, 'Outcome', diabetes_features)

# 2️⃣ Heart
heart = pd.read_csv("heart.csv")
possible_targets = ['target', 'num', 'output', 'Target']
heart_target = next((t for t in possible_targets if t in heart.columns), None)
if not heart_target:
    raise ValueError(f"No target column found in heart.csv! Columns: {heart.columns.tolist()}")

heart_features = ['age', 'sex', 'chol', 'oldpeak']
model_heart, scaler_h, heart_features = train_model(heart, heart_target, heart_features)


parkinsons = pd.read_csv("parkinsons.csv")
parkinsons_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR']
model_parkinsons, scaler_p, parkinsons_features = train_model(parkinsons, 'status', parkinsons_features)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    disease = request.form['disease']

    if disease == 'diabetes':
        model, scaler, features = model_diabetes, scaler_d, diabetes_features
    elif disease == 'heart':
        model, scaler, features = model_heart, scaler_h, heart_features
    else:
        model, scaler, features = model_parkinsons, scaler_p, parkinsons_features

    try:
        values = [float(x) for x in request.form['features'].split(',')]
        if len(values) != len(features):
            return jsonify({"error": f"Expected {len(features)} features for {disease.title()}."})

        input_df = pd.DataFrame([values], columns=features)
        scaled = scaler.transform(input_df)

        # 🔥 PROBABILITY-BASED RISK LOGIC (New)
        prob = float(model.predict_proba(scaled)[0][1])

        if prob < 0.40:
            risk = "Low risk"
        elif prob < 0.70:
            risk = "Moderate risk"
        else:
            risk = "High risk"

        pred = model.predict(scaled)[0]
        result = "Positive" if pred == 1 else "Negative"

        return jsonify({
            "prediction": risk + f" of {disease} detected",
            "risk_level": risk,
            "probability": round(prob, 3),
            "disease": disease,
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_advice', methods=['POST'])
def get_advice():
    data = request.json
    disease = data.get("disease")
    result = data.get("result")
    risk = data.get("risk_level")
    probability = data.get("probability")

    try:
        prompt = f"""
You are a medical doctor. Give a short and friendly health advice summary 
based on the patient’s results. Use emojis to make it user-friendly 
but keep it medically accurate.

Disease: {disease}
Result: {result}
Risk Level: {risk}
Model Probability: {probability}

Write **10-15 lines total**. 
Include these sections:

• 1–2 lines explaining what the result means 😊  
• Symptoms to watch for (bullet points with emojis) 👀  
• Lifestyle changes (bullet points with emojis) 🏃‍♂️🥗  
• Diet suggestions 🍎  
• When to seek medical help ⚠️  

Keep sentences short. No long paragraphs.
"""



        model_gemini = genai.GenerativeModel("gemini-2.5-flash")
        response = model_gemini.generate_content(prompt)
        advice = response.text if hasattr(response, 'text') else "No advice generated."
        
        return jsonify({"advice": advice})

    except Exception as e:
        return jsonify({"advice": f"⚠️ AI advice error: {str(e)}"})
    
if __name__ == '__main__':
    app.run(debug=True)
