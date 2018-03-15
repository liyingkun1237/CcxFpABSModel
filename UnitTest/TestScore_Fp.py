"""
测试模型分 ==》 主要测 变量

这一脚本主要为测试FP计算出的变量 123+62+59

测试的思路：
    1.使用DataFrame作为输入 计算出的变量
    2.使用json串作为输入 计算出的变量
"""

# 1. 使用DataFrame作为输入
import pandas as pd

App_antiLogic = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\App_antiLogic_1030.csv', encoding='gbk')

from CcxFpABSModel.fp_apply import fp_apply

App_antiLogic.head(1)
aa = fp_apply().get_fp_apply_var(App_antiLogic.head(1))  # 123

from CcxFpABSModel.fp_base import fp_base

bb = fp_base().get_fp_base_var(App_antiLogic.head(1))  # 62

from CcxFpABSModel.base_addr import base_addr

cc = base_addr(App_antiLogic.head(1)).get_base_var()  # 59

DDINPUT = pd.merge(pd.merge(aa, bb), cc)
# 277964

# 2. 使用json串作为输入
# C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\Test_genJson.py 使用这个脚本生成json串

# 使用postman 请求 查询计算出的变量
import sqlite3
import numpy as np

cx = sqlite3.connect(r'C:\Users\liyin\Desktop\CcxFpABSModel\databases\FPABS.db')
rr = pd.read_sql('select * from FpABS_RAWData;', cx).drop_duplicates().fillna(np.nan)

# 1.输入数据比对
# col = App_antiLogic.columns
# bool_ = (App_antiLogic.head(1) != rr[col]).sum()
# unequalcol = bool_[bool_ == 1].index
# App_antiLogic.head(1)[unequalcol]
# rr[unequalcol]

'''
###发现一个扯淡的问题 存的时候是nan 读出来是None 复现bug
import numpy as np

pd.DataFrame({'bug': [np.nan]}).to_sql('bug', cx)
pd.read_sql('select * from bug;', cx)
# 一个解决的思路，建表语句 确定一个缺失时的默认值
'''


def f_compare2data(data1, data2):
    '''
    比对两个数据集 注意，一般以1数据集作为参照集
    比对思路：1.数据集的shape 2.数据集的列名 3.数据集交集的值是否相等
    :param data1:
    :param data2:
    :return:
    '''
    in_col = set(data1.columns) & set(data2.columns)
    bool_ = (data1 != data2[data1.columns]).sum()
    unequalcol = bool_[bool_ == 1].index

    import warnings
    if data1.shape != data2.shape:
        warnings.warn('数据集的shape不一致')
    if len(data1.columns) != len(in_col):
        warnings.warn('数据集的列名不完全相等,少的列为{}'.format(set(data1.columns) - in_col))
    elif len(unequalcol) > 0:
        s1 = data1[unequalcol]
        s2 = data2[unequalcol]
        ss = pd.concat([s1, s2])
        # s3 = (s1.dtypes, s2.dtypes)
        warnings.warn('如下列的值不相等\t{},具体如下：\n{}'.format(unequalcol, ss))


f_compare2data(App_antiLogic.head(1), rr)

App_antiLogic.head(1).is_true_home_city == rr.is_true_home_city
# 复现bug np.nan==np.nan



###################比较计算的变量
DDINPUT
vv = pd.read_sql('select * from FPALLVar;', cx).drop_duplicates()  # .fillna(np.nan)

f_compare2data(DDINPUT, vv)

##校验完毕，凡普的使用DataFrame输入和json输入，数据和变量都相等


##########################三方征信数据，直接看变量，因为原始数据实在太多了
from CcxFpABSModel.fp_ccx_process import Fp_ccx_dataprocess

ccx_data = Fp_ccx_dataprocess()



ccx = pd.read_sql('select * from FPALLVar;', cx).drop_duplicates().fillna(np.nan)

ccx_col=['lend_request_id','interval_date_guaduation','FR_REG','NFR_REG',
'credit_code_count','SH_ENT_count','AMT_SUBCONAM','AMT_ACTCONAM',
'sub_act_diff','shixing_times','zhixing_vague_times',
'zhixing_vague_meanAmount','zhixing_vague_maxAmount',
'zhixing_exact_times','zhixing_exact_meanAmount','zhixing_exact_maxAmount',
'overdue_cuiqian_times','overdue_cuiqian_meanAmount','bank_overdue_times',
'xiaodaiOverdue_times','xiaodaiOver_daymax','is_info_leakage','is_directors_of_high',
'verf_times','verf_institution_sum','verf_is_success','verf_is_failed',
'card_is_success','operation_is_success','cid_is_success','bank_verf_times',
'bank_verf_sucess','consumerfin_verf_times','consumerfin_verf_sucess',
'oth_institution_verf_times','oth_institution_verf_sucess','score_query_times',
'score_pass','score_institution_sum','education_type_1.0','education_type_2.0',
'education_type_3.0','education_type_4.0','education_type_5.0','education_type_6.0',
'education_type_nan','educational_1.0','educational_2.0','educational_3.0',
'educational_4.0','educational_nan','guaduation_status_1.0','guaduation_status_2.0',
'guaduation_status_nan','shixing_status_1.0','shixing_status_2.0',
'shixing_status_nan','risk_times']

ccx[ccx_col].to_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\wc_C\onehot_data\277964_var.csv',index=False)



cx_1 = sqlite3.connect(r'C:\Users\liyin\Desktop\CcxFpABSModel\databases\FPABS_4.db')

ccx = pd.read_sql('select * from FPALLVar;', cx_1).drop_duplicates().fillna(np.nan)
ccx[ccx_col].to_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\wc_C\onehot_data\272724_var.csv',index=False)
ccx.columns

ccx.shixing_times
ccx.zhixing_vague_times
