
#--------------------------------------------Import--------------------------------------------#
import pandas as pd
import numpy as np
import pickle

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

import tkinter as tk
from tkinter import messagebox

#--------------------------------------------Featur_Engineering--------------------------------------------#



def choose_mode():
    root = tk.Tk()
    root.withdraw() # Скрываем основное маленькое окно
    
    # Создаем окно с вопросом
    # Возвращает True (Да) или False (Нет)
    result = messagebox.askyesnocancel(
        "Вывести отчет по обучению?", 
        "Нажмите 'Да': Обучение и Отчет \n"
        "Нажмите 'Нет': Обучение и Сохранение (Dump)\n"
        "Нажмите 'Отмена' для выхода"
    )
    
    root.destroy()
    return result


class My_Featur_Engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass 
    
    def fit(self, X, y=None):
        return self  

    def transform(self, X):
        X['utilization_1'] = X['BILL_AMT1'] / X['LIMIT_BAL']
        X['utilization_2'] = X['BILL_AMT2'] / X['LIMIT_BAL']
        X['utilization_3'] = X['BILL_AMT3'] / X['LIMIT_BAL']
        X['utilization_4'] = X['BILL_AMT4'] / X['LIMIT_BAL']
        X['utilization_5'] = X['BILL_AMT5'] / X['LIMIT_BAL']
        X['utilization_6'] = X['BILL_AMT6'] / X['LIMIT_BAL']

        #сумма платежа к прошлому счету 
        X['pay_to_bill_ratio_1'] = X['PAY_AMT1'] / (X['BILL_AMT2'] + 1)
        X['pay_to_bill_ratio_2'] = X['PAY_AMT2'] / (X['BILL_AMT3'] + 1)
        X['pay_to_bill_ratio_3'] = X['PAY_AMT3'] / (X['BILL_AMT4'] + 1)
        X['pay_to_bill_ratio_4'] = X['PAY_AMT4'] / (X['BILL_AMT5'] + 1)
        X['pay_to_bill_ratio_5'] = X['PAY_AMT5'] / (X['BILL_AMT6'] + 1)

        #прирост долга 
        X['bill_diff_1_2'] = X['BILL_AMT1'] - X['BILL_AMT2']
        X['bill_diff_2_3'] = X['BILL_AMT2'] - X['BILL_AMT3']
        X['bill_diff_3_4'] = X['BILL_AMT3'] - X['BILL_AMT4']
        X['bill_diff_4_5'] = X['BILL_AMT4'] - X['BILL_AMT5']
        X['bill_diff_5_6'] = X['BILL_AMT5'] - X['BILL_AMT6']

        # Список всех колонок со счетами
        bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

        # Средний счет и средний платеж
        X['avg_bill'] = X[bill_cols].mean(axis=1)
        X['avg_pay'] = X[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)

        pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

        # Считаем количество месяцев с просрочкой
        X['months_with_delay'] = (X[pay_status_cols] > 0).sum(axis=1)
        
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        X = X.drop(columns=['ID'])

        return X

def load_mmodel(name):
    # Десериализуем pipeline из файла
    with open('name', 'rb') as pkl_file:
        loaded_pipe = pickle.load(pkl_file)


def dump_model(name, model):
    # Сериализуем pipeline и записываем результат в файл
    with open(name, 'wb') as output:
        pickle.dump(model, output)

        

def train_model(X, y):
    
    lo_pipe = Pipeline([  
    ('FeatureEngineering', My_Featur_Engineering()),
    ('FeatureSelection', SelectKBest(f_regression, k=5)),
    ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42))
    ])

    # Обучаем пайплайн
    lo_pipe.fit(X, y)
    
    return lo_pipe


def my_classification_report(model, X, y):
    predictions = model.predict(X)
    print("Отчет по классификации:")
    print(classification_report(y, predictions)) 


def read_df(name='models/UCI_Credit_Card.csv'):
    return pd.read_csv(name)


#--------------------------------------------Fit model --------------------------------------------#
def main_with_report():
    go_train = read_df()

    X = go_train.drop(columns='default.payment.next.month')
    y = go_train['default.payment.next.month']

    go_pipe = train_model(X, y)

    my_classification_report(go_pipe, X, y)


def main_with_dump_model():
    go_train = read_df()

    X = go_train.drop(columns='default.payment.next.month')
    y = go_train['default.payment.next.month']

    gv_name = 'models/model_v1.pkl'

    go_pipe = train_model(X, y)

    dump_model(gv_name,go_pipe)
    

if __name__ == "__main__":
    mode = choose_mode()
    
    if mode is None: #  Отмена
        print("Действие отменено пользователем.")

    if mode is True:
        main_with_report()

    if mode is False:
        main_with_dump_model()






    


    