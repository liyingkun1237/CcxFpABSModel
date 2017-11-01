import pickle
import xgboost as xgb
import numpy as np
import os


def get_model_path(model_name):
    file_path = os.path.split(os.path.realpath(__file__))[0]
    path = os.path.join(file_path, 'exData', model_name)
    return path


def load_model(path):
    with open(path, 'rb') as f:
        bst = pickle.load(f)
    return bst

def get_model_feature_names(bst):
    return bst.feature_names

def model_data(test):
    dtest = xgb.DMatrix(test, missing=np.nan)
    return dtest


def predict_score(dtest, bst):
    test_pred = bst.predict(dtest)
    return test_pred


def score(test_pred):
    score = 600 - 20 / np.log(2) * np.log(test_pred / (1 - test_pred))
    return np.round(score, 0)

