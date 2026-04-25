from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

#from models.model_trian import load_model

from transformers import My_Featur_Engineering

app = Flask(__name__)

# Загрузка модели при старте
#model = load_model('models/model_v1.pkl') - так ругается на методы модели

with open('models/model_v2.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """Эндпоинт для предсказания дефолта"""
    try:
        data = request.get_json()
        features = preprocess_input(data)
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'model_version': 'v2'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Проверка здоровья сервиса"""
    return jsonify({'status': 'healthy'}), 200

def preprocess_input(data):
    """Предобработка входных данных"""
    # Преобразование JSON в numpy array 

    features = pd.DataFrame([data[0]])

    features = features.astype({
        'ID'       : 'int64',
        'LIMIT_BAL': 'float64',
        'SEX'      : 'int64',
        'EDUCATION': 'float64',
        'MARRIAGE' : 'int64',
        'AGE'      : 'float64',
        'PAY_0'    : 'float64',
        'PAY_2'    : 'float64',
        'PAY_3'    : 'float64',
        'PAY_4'    : 'float64',
        'PAY_5'    : 'float64',
        'PAY_6'    : 'float64',
        'BILL_AMT1': 'float64',
        'BILL_AMT2': 'float64',
        'BILL_AMT3': 'float64',
        'BILL_AMT4': 'float64',
        'BILL_AMT5': 'float64',
        'BILL_AMT6': 'float64',
        'PAY_AMT1' : 'float64',
        'PAY_AMT2' : 'float64',
        'PAY_AMT3' : 'float64',
        'PAY_AMT4' : 'float64',
        'PAY_AMT5' : 'float64',         
        'PAY_AMT6' : 'float64',
    })
    
    
    return features

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
