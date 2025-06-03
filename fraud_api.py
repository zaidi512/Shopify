
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model_data = joblib.load('fraud_risk_model_oversampled.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']
risk_encoder = model_data['risk_encoder']
features = model_data['features']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        df = pd.DataFrame([input_data])

        # Feature engineering
        df["Net Amount"] = df["Total"] - df["Discount Amount"] - df["Taxes"]
        df["Free Shipping"] = (df["Shipping"] == 0).astype(int)

        # Encode categorical fields
        for col in ["Payment Method", "Currency"]:
            df[col] = label_encoders[col].transform(df[col])

        # Predict
        prediction = model.predict(df[features])[0]
        label = risk_encoder.inverse_transform([prediction])[0]

        return jsonify({"fraud_risk": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
