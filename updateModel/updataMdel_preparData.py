"""
准备数据

"""
import pandas as pd

from CcxFpABSModel.calMain import f_calMain
from CcxFpABSModel.fp_ccx_process import Fp_ccx_dataprocess

education = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\education_all.csv', encoding='gbk')
gaoguan = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\gaoguan.csv', encoding='gbk')
gudong = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\gudong.csv', encoding='gbk')
shixing = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\shixing.csv', encoding='gbk')
zhixing_vague = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\zhixing_vague.csv',
                            encoding='gbk')
zhixing_exact = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\zhixing_exact.csv',
                            encoding='gbk')
overdue_cuiqian = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\overdue_cuiqian.csv',
                              encoding='gbk')
bank_ovredue = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\bank_ovredue.csv',
                           encoding='gbk')
xiaodai_overdue = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\xiaodai_overdue.csv',
                              encoding='gbk')
info_leakage = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\info_leakage.csv',
                           encoding='gbk')
cid_verf = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\cid_verf.csv', encoding='gbk')
mob_verf = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\mob_verf.csv', encoding='gbk')
cid_score = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\cid_score.csv', encoding='gbk')
mob_score = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\mob_score.csv', encoding='gbk')

######################################
"""education 是有全的key的"""
ccx_Rawdata = {"education": education, "gaoguan": gaoguan,
               "gudong": gudong, "shixin": shixing,
               "zhixing_vague": zhixing_vague,
               "zhixing_exact": zhixing_exact,
               "overdue_cuiqian": overdue_cuiqian,
               "bank_ovredue": bank_ovredue, "xiaodai_overdue": xiaodai_overdue,
               "info_leakage": info_leakage, "cid_verf": cid_verf,
               "mob_verf": mob_verf, "cid_score": cid_score, "mob_score": mob_score
               }

ccx_data = Fp_ccx_dataprocess()  # 生成对象
var_list = [ccx_data.education_data(ccx_Rawdata['education']),
            ccx_data.senior_data(ccx_Rawdata['gaoguan']),
            ccx_data.shareholder_data(ccx_Rawdata['gudong']),
            ccx_data.lose_promise_data(ccx_Rawdata['shixin']),  # shixing --> shixin 1104更改
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

##########
'''
python merge
'''
fp_data = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\App_antiLogic_1030.csv', encoding='gbk')

FPALLVAR = f_calMain(fp_data, ccx_Rawdata).drop_duplicates('lend_request_id')

FPALLVAR.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\FPALLVAR_1105.csv', index=False)

ccx_data.execute_vague_data(ccx_Rawdata['zhixing_vague']).query('lend_request_id==323871')
ccx_data.lose_promise_data(ccx_Rawdata['shixin']).query('lend_request_id==345953')


##############
r"""
11月5号，进行评分修正时发现的bug点
def lose_promise_data(data):
    data=data.copy()
    data['立案时间失信'] = data['立案时间失信'].replace(
        ['40907', '41623', '41391', '41275', '41302', '41453', '41456', '41040', '41142'], '2015年11月11日')
    data['立案时间失信'] = data['立案时间失信'].apply(Fp_ccx_dataprocess().f_transdate)
    data['PassMth'] = pd.to_datetime(data['PassMth'])
    data = data.loc[data['立案时间失信'] < data['PassMth']]
    df = data.groupby('lend_request_id')['被执行人履行情况'].agg(['count', np.max]).reset_index().rename(
        columns={'count': 'shixing_times', 'amax': 'shixing_status'})
    return df
data = shixing
shixing['立案时间失信'].value_counts()
data['立案时间失信']
shixing = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\ccx_rawdata\shixing.csv', encoding='gbk')
lose_promise_data(shixing)
shixing.shape
shixing['立案时间失信'].apply(len).value_counts()

"""

####### 发现bug后，需重跑程序 和重新更新模型
# fp_ccx_data_dummy.shixing_times.value_counts()
# FPALLVAR.shixing_times.value_counts()