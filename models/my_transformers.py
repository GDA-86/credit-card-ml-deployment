import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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