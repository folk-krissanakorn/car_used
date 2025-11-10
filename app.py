from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime

app = Flask(__name__)

# โหลดโมเดลและ scaler
model = joblib.load('best_car_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# ✅ รายการ feature ที่โมเดลใช้
FEATURES = ['milage_km', 'car_age', 
    'brand_Toyota', 'brand_Honda', 'brand_Mazda', 'brand_Nissan', 'brand_Mitsubishi',
    'fuel_type_Gasoline', 'fuel_type_Hybrid', 'fuel_type_Electric', 'fuel_type_Diesel'
]

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

        # 🔹 คำนวณอายุรถ
        current_year = datetime.datetime.now().year
        car_age = current_year - year

        # 🔹 เตรียม one-hot encoding ให้ตรงกับโมเดล
        input_dict = {col: 0 for col in FEATURES}
        input_dict['milage_km'] = milage
        input_dict['car_age'] = car_age

        # Brand
        brand_col = f"brand_{brand}"
        if brand_col in input_dict:
            input_dict[brand_col] = 1

        # Fuel
        fuel_col = f"fuel_type_{fuel}"
        if fuel_col in input_dict:
            input_dict[fuel_col] = 1

        # 🔹 แปลงเป็น array
        input_array = np.array([input_dict[col] for col in FEATURES]).reshape(1, -1)

        # scale ข้อมูล
        input_scaled = scaler.transform(input_array)

        # 🔹 ทำนาย
        predicted_price = model.predict(input_scaled)[0]
        price_thb = round(predicted_price, 2)

        return jsonify({
            "predicted_price_thb": price_thb,
            "brand": brand,
            "model": model_name,
            "year": year,
            "fuel": fuel
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
