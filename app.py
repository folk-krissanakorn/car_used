from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# โหลดโมเดลและ scaler
model = joblib.load('best_car_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "🚗 Car Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # รับข้อมูลจากผู้ใช้ (JSON)
        data = request.get_json()

        # ตัวอย่าง feature ที่ต้องป้อน
        # ['milage_km', 'car_age', 'brand_Toyota', 'fuel_type_Gasoline', ...]
        input_data = np.array(data['features']).reshape(1, -1)

        # scale ข้อมูลก่อนส่งเข้าโมเดล
        input_scaled = scaler.transform(input_data)

        # ทำนายราคา
        prediction = model.predict(input_scaled)
        price = round(prediction[0], 2)

        return jsonify({'predicted_price_thb': price})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
