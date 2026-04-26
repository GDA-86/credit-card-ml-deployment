from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

#from models.model_trian import load_model

from my_transformers import My_Featur_Engineering

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


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_', methods=['GET'])
def predict_get():
    """Проверка через get """

    #http://localhost:5000/predict?ID=0&LIMIT_BAL=70000&SEX=2&EDUCATION=2&MARRIAGE=2&AGE=26&PAY_0=2
    # &PAY_2=0&PAY_3=0&PAY_4=2&PAY_5=2&PAY_6=2&BILL_AMT1=41087&BILL_AMT2=42445&BILL_AMT3=45020&BILL_AMT4=44006
    # &BILL_AMT5=46905&BILL_AMT6=46012&PAY_AMT1=2007&PAY_AMT2=3582&PAY_AMT3=0&PAY_AMT4=3601&PAY_AMT5=0&PAY_AMT6=1820
    try:
        data = request.args.to_dict()
        
        for key in data:
            try:
                data[key] = float(data[key]) # type: ignore
            except ValueError:
                pass
    except Exception as e:
        return jsonify({'error': str(e)}), 400


    features = pd.DataFrame([data])

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    
    return jsonify({'received_data': data, 
                    'status': 'ok',
                    'prediction': int(prediction[0]),
                    'probability' : float(probability),
                    'model_version': 'v2'
                    })
    


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
