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
