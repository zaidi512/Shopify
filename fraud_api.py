from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
import os

app = Flask(__name__)

# Load model
model_data = joblib.load('fraud_risk_model_oversampled.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']
risk_encoder = model_data['risk_encoder']
features = model_data['features']

# Shopify API credentials
SHOPIFY_API_KEY = "cd014c001d7a15705c088a24a2d97e9a"
SHOPIFY_PASSWORD = "Hassan@512"
SHOPIFY_STORE = "br0d07-7a.myshopify.com"  # No trailing slash

@app.route('/')
def home():
    return "✅ Fraud Detection API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        df = pd.DataFrame([input_data])
        df["Net Amount"] = df["Total"] - df["Discount Amount"] - df["Taxes"]
        df["Free Shipping"] = (df["Shipping"] == 0).astype(int)

        # Encode categorical fields
        for col in ["Payment Method", "Currency"]:
            df[col] = df[col].apply(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
            df[col] = label_encoders[col].transform(df[col])

        prediction = model.predict(df[features])[0]
        label = risk_encoder.inverse_transform([prediction])[0]

        return jsonify({"fraud_risk": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/shopify_webhook', methods=['POST'])
def shopify_webhook():
    print("✅ Webhook received")

    try:
        order = request.json
        print("🛒 Order payload:", order)

        # Handle missing fields gracefully for test webhooks
        order_id = order.get("id")
        if not order_id:
            print("❌ Missing order ID in webhook payload.")
            return jsonify({"error": "Missing order ID"}), 400

        shipping_price = 0.0
        if "shipping_lines" in order and order["shipping_lines"]:
            shipping_price = float(order["shipping_lines"][0].get("price", 0))

        payment_method = order.get("payment_gateway_names", ["Unknown"])[0]

        input_data = {
            "Total": float(order.get("total_price", 0)),
            "Shipping": shipping_price,
            "Taxes": float(order.get("total_tax", 0)),
            "Discount Amount": float(order.get("total_discounts", 0)),
            "Payment Method": payment_method,
            "Currency": order.get("currency", "USD")
        }

        print("📦 Processed Input:", input_data)

        # Feature engineering
        df = pd.DataFrame([input_data])
        df["Net Amount"] = df["Total"] - df["Discount Amount"] - df["Taxes"]
        df["Free Shipping"] = (df["Shipping"] == 0).astype(int)

        # Handle unknown categories
        for col in ["Payment Method", "Currency"]:
            known_classes = label_encoders[col].classes_
            df[col] = df[col].apply(lambda x: x if x in known_classes else known_classes[0])
            df[col] = label_encoders[col].transform(df[col])

        prediction = model.predict(df[features])[0]
        risk_label = risk_encoder.inverse_transform([prediction])[0]

        print(f"🛡️ Predicted Risk for Order {order_id}: {risk_label}")

        return jsonify({"status": "success", "fraud_risk": risk_label})

    except Exception as e:
        print("❌ Exception occurred:", e)
        return jsonify({"error": str(e)}), 400


def tag_order(order_id, fraud_label):
    get_url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_PASSWORD}@{SHOPIFY_STORE}/admin/api/2023-10/orders/{order_id}.json"
    response = requests.get(get_url)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch order {order_id}")
        return

    existing_tags = response.json().get("order", {}).get("tags", "")
    updated_tags = existing_tags + f",Fraud-{fraud_label}"

    put_url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_PASSWORD}@{SHOPIFY_STORE}/admin/api/2023-10/orders/{order_id}.json"
    headers = { "Content-Type": "application/json" }
    data = { "order": { "id": order_id, "tags": updated_tags.strip(',') } }

    put_response = requests.put(put_url, json=data, headers=headers)
    if put_response.status_code != 200:
        print(f"❌ Failed to update tags for order {order_id}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
