
#--------------------------------------------Import--------------------------------------------#
import pandas as pd

import pickle

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier

import tkinter as tk
from tkinter import messagebox


from transformers import My_Featur_Engineering

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




def load_model(name):
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






    


    