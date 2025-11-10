from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, origins=["https://carfront-iota.vercel.app", "*"])  # ✅ รองรับ CORS

# โหลดโมเดลและ scaler
model = joblib.load('best_car_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "🚗 Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        brand = data.get('brand')
        model_name = data.get('model')
        year = int(data.get('year'))
        milage = float(data.get('milage'))
        fuel = data.get('fuel')

        # ✅ ตัวอย่างการแปลงข้อมูล (คุณสามารถแก้ให้ตรงกับโมเดลของคุณได้)
        car_age = 2025 - year

        # ตัวอย่าง features (ปรับตามโมเดลจริง)
        # สมมติว่าคุณ train ด้วย [milage, car_age]
        features = np.array([[milage, car_age]])
        input_scaled = scaler.transform(features)

        # ทำนายราคา
        prediction = model.predict(input_scaled)
        price = round(prediction[0], 2)

        return jsonify({'predicted_price_thb': price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
