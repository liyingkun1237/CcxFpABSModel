"""
当一次请求完成之后 异步的开启一个存储过程 将计算的数据存储下来

存储的表名为：
FpABS_RAWData
FPALLVar
FPResult

三方数据表名较多
"""

from CcxFpABSModel.config import FPABSDATABASEPATH, FPABSDATABASE
import pandas as pd
import os
import sqlite3

from CcxFpABSModel.log import ABS_log


@ABS_log('FPABS')
def f_SaveDate(rawdata, tablename):
    '''
    https://www.2cto.com/database/201606/513837.html 参考网站
    :param rawdata: 数据集
    :param filename: sqlite中，数据库中的表名
    :return:
    '''

    if os.path.exists(FPABSDATABASEPATH):  # 如果存放数据库的路径不存在 则创建
        with sqlite3.connect(FPABSDATABASE) as conn:
            rawdata.to_sql(tablename, conn, index=False, if_exists='append')
    else:
        os.mkdir(FPABSDATABASEPATH)
        with sqlite3.connect(FPABSDATABASE) as conn:
            rawdata.to_sql(tablename, conn, index=False, if_exists='append')

            # print('数据存储成功')


def f_threadSaveData(fp_data, ccx_Rawdata):
    '''

    :param fp_data: 凡普的数据
    :param ccx_Rawdata: 三方征信数据 {'数据集':DataFrame}
    :return:
    '''
    # print(fp_data.is_true_home_city)
    f_SaveDate(fp_data, 'FpABS_RAWData')
    for k in ccx_Rawdata.keys():
        # print('#' * 32, k, ccx_Rawdata[k], )
        f_SaveDate(ccx_Rawdata[k], k)

    print('所有数据保存成功')


def f_threadSaveVar(var):
    '''

    :param var: 模型计算出来的变量
    :return:
    '''
    f_SaveDate(var, 'FPALLVar')
    print('变量保存成功')


def f_threadSaveResult(res):
    '''
    保存返回的结果
    :param res:
    :return:
    '''
    f_SaveDate(pd.DataFrame(res, index=[0]), 'FPResult')
    print('结果保存成功')


@ABS_log('FPABS')
def f_threadSave(fp_data, ccx_Rawdata, var, res):
    f_threadSaveData(fp_data, ccx_Rawdata)
    f_threadSaveVar(var)
    f_threadSaveResult(res)


if __name__ == '__main__':
    cx = sqlite3.connect(r'C:\Users\liyin\Desktop\CcxFpABSModel\databases\FPABS.db')
    # bankcard_dict.head(5).to_sql('fp', cx, index=False, if_exists='append')
    # append 会出现重复数据 可优化为insertover
    pd.read_sql('select * from bank;', cx)
