from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ เพิ่มบรรทัดนี้
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ เปิดให้ทุก origin เข้าถึง API ได้
# หรือถ้าจะจำกัดเฉพาะ frontend ของคุณ
# CORS(app, origins=["https://carfront-iota.vercel.app"])

# โหลดโมเดลและ scaler
model = joblib.load('best_car_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "🚗 Car Price Prediction API is running with CORS enabled!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        mileage = float(data.get("mileage", 0))
        car_age = float(data.get("car_age", 0))
        brand = data.get("brand", "")
        model_name = data.get("model", "")
        fuel = data.get("fuel", "")

        brands = ["Toyota", "Honda", "Mazda", "Nissan"]
        models = ["Vios", "Camry", "Altis", "Civic", "City", "Accord", "Mazda2", "Mazda3", "CX-5", "Almera", "Navara", "Teana"]
        fuels = ["Gasoline", "Diesel", "Hybrid", "EV"]

        brand_encoded = [1 if brand == b else 0 for b in brands]
        model_encoded = [1 if model_name == m else 0 for m in models]
        fuel_encoded = [1 if fuel == f else 0 for f in fuels]

        features = [mileage, car_age] + brand_encoded + model_encoded + fuel_encoded
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)
        price = round(prediction[0], 2)

        return jsonify({'predicted_price_thb': price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
