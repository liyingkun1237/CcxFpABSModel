import pickle
import xgboost as xgb
import numpy as np
import os

from CcxFpABSModel.log import ABS_log


@ABS_log('FPABS')
def get_model_path(model_name):
    file_path = os.path.split(os.path.realpath(__file__))[0]
    path = os.path.join(file_path, 'exData', model_name)
    return path


@ABS_log('FPABS')
def load_model(path):
    with open(path, 'rb') as f:
        bst = pickle.load(f)
    return bst


def get_model_feature_names(bst):
    return bst.feature_names


@ABS_log('FPABS')
def model_data(test):
    dtest = xgb.DMatrix(test, missing=np.nan)
    return dtest


@ABS_log('FPABS')
def predict_score(dtest, bst):
    test_pred = bst.predict(dtest)
    return test_pred


@ABS_log('FPABS')
def score(test_pred):
    score = 600 - 20 / np.log(2) * np.log(test_pred / (1 - test_pred))
    return np.round(score, 0)


# @ABS_log('FPABS')
def f_mdscore(res, VAR):
    '''

    :param res: 预测出的评分 返回格式为字典
    :param VAR: 计算出的全部变量
    :return: 修正后的得分
    '''

    def rule(presocre, value):
        ls = []
        for i, j in zip(presocre, value):
            if i > 670 and j > 0:  # 满足入池条件且有风险
                ls.append(i - (i - 670) * j * 1.5)
            else:
                ls.append(i)
        return ls

    value = list(VAR.shixing_times.fillna(0) + VAR.zhixing_exact_times.fillna(0))
    return rule([res['Ccx_score']], value)[0]


# @ABS_log('FPABS')
def f_mdscorebyRisk(new_pre_score, ccx_Rawdata):
    riskscore = ccx_Rawdata['riskScore']['riskScore'].values[0]
    newsocre = new_pre_score - riskscore  # 经过修正的模型分 减去风险分
    if new_pre_score > 670 and newsocre > 670 and riskscore > 30:  # 即修正前 修正后 且有风险
        '''
        riskscore > 30 特别注意这个条件 没有数据核验过 该减多少
        减30的原因，线下最高评分也就700分，修正后，还超过670，说明了风险分超了30，且最初评分很高
        '''
        return newsocre - (newsocre - 670) * 2
    else:
        return newsocre
