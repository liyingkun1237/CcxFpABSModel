import multiprocessing

import flask
from flask import request
import json
import pandas as pd
from datetime import datetime
from flask import jsonify

from CcxFpABSModel.SaveDate import f_threadSave, f_threadSaveScore
from CcxFpABSModel.calMain import f_calMain
from CcxFpABSModel.model import *
from CcxFpABSModel.updateBankDict import f_updateBankDict
import time
from CcxFpABSModel.log import ABS_log
import threading

server = flask.Flask(__name__)


@server.route('/ccxfpABSModelApi', methods=['post'])
def ccxfpABSModelApi():
    try:
        st = time.time()
        # 1.获取数据
        rawdata = json.loads(request.data.decode())
        reqID = rawdata.get('reqID')  # 获取请求ID，建议为进件号，便于查错
        fp_data = pd.DataFrame(rawdata.get('FpData')).fillna(np.nan)
        ccx_Rawdata = f_ccxDataTransform(rawdata.get("CcxData"))

        # 创建bank
        f_updateBankDict(ccx_Rawdata['bank'])

        # 2.计算变量
        VAR = f_calMain(fp_data, ccx_Rawdata)

        # 4.预测评分
        VAR.index = VAR.lend_request_id
        # VAR.drop(['lend_request_id'], inplace=True)
        model_path = get_model_path('model_Fp_Ccx_All_2017-11-05.txt')
        bst = load_model(model_path)
        pvalue = predict_score(model_data(VAR[bst.feature_names]), bst)
        pre_score = score(pvalue).tolist()[0]
        # 5. 返回结果
        curDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        res = {'code': '200', 'code_message': "计算成功", 'pvalue': str(pvalue.tolist()[0]), 'Ccx_score': pre_score,
               "reqTime": curDate, "reqID": str(reqID)
               }

        # 6. 修正评分 使用shixing_times zhixing_exact_times
        new_pre_score = f_mdscore(res, VAR)

        # 7. 修正评分 使用风险分
        new_score = f_mdscorebyRisk(new_pre_score, ccx_Rawdata)

        res['Ccx_score'] = str(new_score)
        # print(request.data.decode())
        # print('计算用时:', time.time() - st)
        res_score = {'reqID': str(reqID), 'preScore': pre_score, 'mdScore': new_pre_score,
                     'mdScorebyRisk': new_score}
        # f_threadSaveScore(res_score)
        print('INFO -->> %s' % res_score)

        # 起一个异步线程去存储评分结果
        with server.app_context():
            t = multiprocessing.Process(target=f_threadSaveScore, args=(res_score,))
            t.start()

        # 起一个异步线程去存储数据
        with server.app_context():
            t = multiprocessing.Process(target=f_threadSave, args=(fp_data, ccx_Rawdata, VAR, res))
            t.start()
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        res = {"code": "502", "msg": "计算失败", "error_msg": str(e), "reqID": str(reqID)}
        print('ERROR -->> %s' % res)
        return jsonify(res)


@ABS_log('FPABS')
def f_ccxDataTransform(data_dict):
    new_data_dict = {}
    data_dict = dict(data_dict)
    for k in data_dict.keys():
        try:
            new_data_dict[k] = pd.DataFrame(data_dict[k]).fillna(np.nan).replace("null", np.nan)  # 将None替换为Nan
        except:
            new_data_dict[k] = pd.DataFrame()
            # 写日志了
    return new_data_dict


if __name__ == '__main__':
    server.run(debug=True, port=5051, host='0.0.0.0', processes=100)
