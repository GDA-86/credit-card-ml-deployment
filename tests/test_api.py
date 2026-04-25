import requests
import pandas as pd



if __name__ == '__main__':
    # выполняем POST-запрос на сервер по эндпоинту add с параметром json

    #Порядок соответсвует обученному DF 

    #1,20000,2,2,1,24,2,2,-1,-1,-2,-2,3913,3102,689,0,0,0,0,689,0,0,0,0,
    #23,70000,2,2,2,26,2,0,0,2,2,2,41087,42445,45020,44006,46905,46012,2007,3582,0,3601,0,1820,1
    data_dict = [{
        "ID": 0,
        "LIMIT_BAL": 70000.0,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 2,
        "AGE": 26,
        "PAY_0": 2,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 2,
        "PAY_5": 2,
        "PAY_6": 2,
        "BILL_AMT1": 41087.0,
        "BILL_AMT2": 42445.0,
        "BILL_AMT3": 45020.0,
        "BILL_AMT4": 44006.0,
        "BILL_AMT5": 46905.0,
        "BILL_AMT6": 46012.0,
        "PAY_AMT1": 2007.0,
        "PAY_AMT2": 3582.0,
        "PAY_AMT3": 0.0,
        "PAY_AMT4": 3601.0,
        "PAY_AMT5": 0.0,
        "PAY_AMT6": 1820.0
    }]
    
    
    r = requests.post('http://localhost:5000/predict', json=data_dict)
    # выводим статус запроса
    print('Status code: {}'.format(r.status_code))
    # реализуем обработку результата
    if r.status_code == 200:
        # если запрос выполнен успешно (код обработки=200),
        # выводим результат на экран
        print('Prediction: {}'.format(r.json()['prediction']))
    else:
        # если запрос завершён с кодом, отличным от 200,
        # выводим содержимое ответа
        print(r.text)