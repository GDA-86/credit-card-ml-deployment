def my_Featur_Engineering(df):
    df['utilization_1'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    df['utilization_2'] = df['BILL_AMT2'] / df['LIMIT_BAL']
    df['utilization_3'] = df['BILL_AMT3'] / df['LIMIT_BAL']
    df['utilization_4'] = df['BILL_AMT4'] / df['LIMIT_BAL']
    df['utilization_5'] = df['BILL_AMT5'] / df['LIMIT_BAL']
    df['utilization_6'] = df['BILL_AMT6'] / df['LIMIT_BAL']

    #сумма платежа к прошлому счету 
    df['pay_to_bill_ratio_1'] = df['PAY_AMT1'] / (df['BILL_AMT2'] + 1)
    df['pay_to_bill_ratio_2'] = df['PAY_AMT2'] / (df['BILL_AMT3'] + 1)
    df['pay_to_bill_ratio_3'] = df['PAY_AMT3'] / (df['BILL_AMT4'] + 1)
    df['pay_to_bill_ratio_4'] = df['PAY_AMT4'] / (df['BILL_AMT5'] + 1)
    df['pay_to_bill_ratio_5'] = df['PAY_AMT5'] / (df['BILL_AMT6'] + 1)

    #прирост долга 
    df['bill_diff_1_2'] = df['BILL_AMT1'] - df['BILL_AMT2']
    df['bill_diff_2_3'] = df['BILL_AMT2'] - df['BILL_AMT3']
    df['bill_diff_3_4'] = df['BILL_AMT3'] - df['BILL_AMT4']
    df['bill_diff_4_5'] = df['BILL_AMT4'] - df['BILL_AMT5']
    df['bill_diff_5_6'] = df['BILL_AMT5'] - df['BILL_AMT6']

    # Список всех колонок со счетами
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']

    # Средний счет и средний платеж
    df['avg_bill'] = df[bill_cols].mean(axis=1)
    df['avg_pay'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)

    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Считаем количество месяцев с просрочкой
    df['months_with_delay'] = (df[pay_status_cols] > 0).sum(axis=1)


    df = df.drop(columns=['ID'])

    return df



