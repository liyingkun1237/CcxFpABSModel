"""
2017 - 11 -30 为马雪君提供数据

数据要求
"""

import pandas as pd

score = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\rule_1112.csv')

['lend_request_id', 'PassMth', 'name', 'mobile', 'id_no', 'card_no_pri']

lendid_14_17 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\凡普进件号（历史数据+入池增量数据）.xlsx', 0)
lendid_14_17 = lendid_14_17.rename(columns={'进件号': 'lend_request_id'})
lendid_17_56 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\凡普进件号（历史数据+入池增量数据）.xlsx', 1)
lendid_17_56 = lendid_17_56.rename(columns={'进件号': 'lend_request_id'})


# 2.再去找四要素
lendid_14_17 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\中诚信四要素及PassMth.xlsx', 1)
lendid_17_56 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\中诚信四要素及PassMth.xlsx', 2)

lendid_14_17_score = pd.merge(score[['lend_request_id', 'mdScore']], lendid_14_17)
lendid_17_56_score = pd.merge(score[['lend_request_id', 'mdScore']], lendid_17_56)

lendid_14_17_score.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\lendid_14_17_score.csv',
                          index=False)
lendid_17_56_score.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\lendid_17_56_score.csv',
                          index=False)
