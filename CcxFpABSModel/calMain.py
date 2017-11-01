"""
计算变量的主函数

流程：

1.解析输入的json为DataFrame

2.将新增的银行卡数据更新到银行卡的字典中

3.计算fp_apply;fp_base;base_addr三部分相关的变量

4.计算三方征信相关的变量

5.将四部分数据匹配为总的VAR变量集

6.输入模型中，进行预测 并计算出得分

7.依据风险信息修正得分，并给出是否入池的判别标准

8.返回结果

"""
from CcxFpABSModel.fp_apply import fp_apply
from CcxFpABSModel.fp_base import fp_base
from CcxFpABSModel.base_addr import base_addr
from CcxFpABSModel.fp_ccx_process import Fp_ccx_dataprocess
import pandas as pd
from functools import reduce

from CcxFpABSModel.log import ABS_log


@ABS_log('FPABS')
def f_calMain(fp_data, ccx_Rawdata):
    '''
    计算变量的总函数
    :param fp_data: DataFrame格式
    :param ccx_Rawdata: 字典格式，型如{"education":education['']}
    :return: 变量DataFrame
    '''
    # 3.计算fp_apply;fp_base;base_addr三部分相关的变量
    fpApply_Var = fp_apply().get_fp_apply_var(fp_data)
    fpBase_Var = fp_base().get_fp_base_var(fp_data)
    fpAddr_Var = base_addr(fp_data).get_base_var()

    # 4.计算三方征信相关的变量
    ccx_data = Fp_ccx_dataprocess()  # 生成对象
    var_list = [ccx_data.education_data(ccx_Rawdata['education']),
                ccx_data.senior_data(ccx_Rawdata['gaoguan']),
                ccx_data.shareholder_data(ccx_Rawdata['gudong']),
                ccx_data.lose_promise_data(ccx_Rawdata['shixing']), \
                ccx_data.execute_vague_data(ccx_Rawdata['zhixing_vague']),
                ccx_data.execute_exact_data(ccx_Rawdata['zhixing_exact']),
                ccx_data.overdue_cuiqian_data(ccx_Rawdata['overdue_cuiqian']), \
                ccx_data.bank_overdue_data(ccx_Rawdata['bank_ovredue']),
                ccx_data.xiaodai_overdue_data(ccx_Rawdata['xiaodai_overdue']),
                ccx_data.info_leakage_data(ccx_Rawdata['info_leakage']), \
                ccx_data.is_directors_of_high(ccx_Rawdata['gaoguan'], ccx_Rawdata['gudong']),
                ccx_data.verf_output_data(ccx_Rawdata['cid_verf'], ccx_Rawdata['mob_verf']),
                ccx_data.score_data(ccx_Rawdata['cid_score'], ccx_Rawdata['mob_score'])]

    fp_ccx_data = ccx_data.merge_data(var_list)  # pre_onehot的数据
    fp_ccx_data_dummy = ccx_data.dummy_ccx_data(fp_ccx_data)  # 用于模型的数据 58列

    # 合并四部分计算出的变量
    VAR = reduce(merge_reduce, [fpApply_Var, fpBase_Var, fpAddr_Var, fp_ccx_data_dummy])

    return VAR


def merge_reduce(x, y):
    return pd.merge(x, y, on='lend_request_id', how='left')
