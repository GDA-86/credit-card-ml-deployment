from flask import Flask, request, jsonify
import pickle
import numpy as np

#from models.model_trian import load_model

from transformers import My_Featur_Engineering

app = Flask(__name__)

# Загрузка модели при старте
#model = load_model('models/model_v1.pkl') - так ругается на методы модели

with open('models/model_v1.pkl', 'rb') as f:
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
            'model_version': 'v1'
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
    features = np.array([data[key] for key in sorted(data.keys())]).reshape(1, -1)

    
    return features

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
