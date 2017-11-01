import flask
from flask import request
import json
import pandas as pd
from datetime import datetime
from flask import jsonify
from CcxFpABSModel.calMain import f_calMain
from CcxFpABSModel.model import *

server = flask.Flask(__name__)


@server.route('/ccxfpABSModelApi', methods=['post'])
def ccxfpABSModelApi():
    try:
        # 1.获取数据
        rawdata = json.loads(request.data.decode())
        reqID = rawdata.get('reqID')  # 获取请求ID，建议为进件号，便于查错
        fp_data = pd.DataFrame(rawdata.get('FpData')).fillna(np.nan)
        ccx_Rawdata = f_ccxDataTransform(rawdata.get("CcxData"))

        # 2.计算变量
        VAR = f_calMain(fp_data, ccx_Rawdata)

        # 4.预测评分
        VAR.index = VAR.lend_request_id
        # VAR.drop(['lend_request_id'], inplace=True)
        model_path = get_model_path('model_Fp_Ccx_All_2017-09-29.txt')
        bst = load_model(model_path)
        pvalue = predict_score(model_data(VAR[bst.feature_names]), bst)
        pre_score = score(pvalue).tolist()[0]
        # 5. 返回结果
        curDate = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        res = {'code': 200, 'code_message': "计算成功", 'pvalue': pvalue.tolist()[0], 'Ccx_score': pre_score,
               "reqTime": curDate, "reqID": reqID
               }

        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return jsonify({"code": 502, "msg": "计算失败", "error_msg": str(e), "reqID": reqID})


def f_ccxDataTransform(data_dict):
    new_data_dict = {}
    data_dict = dict(data_dict)
    for k in data_dict.keys():
        try:
            new_data_dict[k] = pd.DataFrame(data_dict[k]).fillna(np.nan)  # 将None替换为Nan
        except:
            new_data_dict[k] = pd.DataFrame()
            # 写日志了
    return new_data_dict


if __name__ == '__main__':
    server.run(debug=True, port=5051, host='0.0.0.0')
