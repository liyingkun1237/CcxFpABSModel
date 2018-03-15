"""
计算1101 新增的6万多数据

"""

import pandas as pd

from CcxFpABSModel.calMain import f_calMain
from CcxFpABSModel.fp_ccx_process import Fp_ccx_dataprocess

education = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\education.csv',
                        encoding='gbk')
gaoguan = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\gaoguan.csv',
                      encoding='gbk')
gudong = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\gudong.csv', encoding='gbk')
shixing = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\shixing.csv',
                      encoding='gbk')
zhixing_vague = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\zhixing_vague.csv',
                            encoding='gbk')
zhixing_exact = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\zhixing_exact.csv',
                            encoding='gbk')
overdue_cuiqian = pd.read_csv(
    r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\overdue_cuiqian.csv',
    encoding='gbk')
bank_ovredue = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\bank_ovredue.csv',
                           encoding='gbk')
xiaodai_overdue = pd.read_csv(
    r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\xiaodai_overdue.csv',
    encoding='gbk')
info_leakage = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\info_leakage.csv',
                           encoding='gbk')
cid_verf = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\cid_verf.txt',
                       encoding='utf-8')
mob_verf = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\mob_verf.txt',
                       encoding='utf-8')
cid_score = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\cid_score.txt',
                        encoding='utf-8')
mob_score = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\ccx_rawdata\mob_score.txt',
                        encoding='utf-8')

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

####凡普的申请资料数据
import pandas as pd

apply14 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\20171031中诚信模型数据\客户字段表（201401-201408）.xlsx')
apply14.shape
# (4101, 61)
apply14.columns

apply1417 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\20171031中诚信模型数据\客户字段表（201409-201705）.xlsx')
apply1417.shape
# (32715, 61)


apply170506 = pd.read_excel(
    r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\20171031中诚信模型数据\客户字段表（201705-201706）.xlsx')
apply170506.shape
# (23679, 61)

apply = pd.concat([apply14, apply1417, apply170506])
apply.shape
# (60495, 61)

apply.drop_duplicates().shape
# (59495, 61)

redict = {'借款人收入': 'Borrower_monthincome',
          '合同期限': 'Contract_period',
          '合同金额': 'Contract_amount',
          '借款人地区': 'Borrower_area',
          '借款人工作单位': 'Work_unit',
          '户籍省份': 'register_prov',
          '户籍城市': 'register_city',
          '居住省份': 'live_prov',
          '居住城市': 'live_city',
          '资产分级': 'asset_grad',
          '收入负债比': 'DIR'
          }
apply = apply.rename(columns=redict)
apply = apply.drop_duplicates()
# 读取四要素数据
f4_1 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\中诚信四要素及PassMth.xlsx', sheetname=0)
f4_2 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\中诚信四要素及PassMth.xlsx', sheetname=1)
f4_3 = pd.read_excel(r'C:\Users\liyin\Desktop\CcxFpABSModel\1101AddData\中诚信四要素及PassMth.xlsx', sheetname=2)

f4 = pd.concat([f4_1, f4_2, f4_3])
f4 = f4.drop_duplicates()

f4.columns
re_dict = {
    'pass_mth': 'PassMth',
    '姓名': 'name',
    '手机号': 'mobile',
    '身份证号': 'id_no',
    '银行卡号': 'card_no_pri'
}
f4 = f4.rename(columns=re_dict)

apply_all = pd.merge(f4, apply)

apply_all.columns
xy_col = ['lend_request_id', 'PassMth', 'name', 'mobile',
          'id_no', 'card_no_pri', 'age', 'annual_income',
          'car_property_type', 'department', 'education', 'gender',
          'have_bro_sis', 'have_children', 'have_colleague', 'have_friends',
          'have_others', 'have_parent', 'have_relatives', 'have_spouse',
          'house_property_type', 'iNatives', 'is_company_native', 'is_exist_children',
          'is_exist_weibo', 'is_exist_weixin', 'is_pay_of_social_security_fund',
          'is_true_home_city', 'is_true_houce_city', 'is_true_idno_city',
          'is_true_income', 'job_title', 'lend_company_type', 'lend_request_id',
          'life_years', 'loan_amount', 'loan_purpose', 'marriage', 'qq_length',
          'resource_type', 'secret_to_family', 'tel_same_city_str', 'tel_same_type_str',
          'work_years', 'DIR', 'Borrower_monthincome', 'Contract_amount',
          'Contract_period', 'Borrower_area', 'Work_unit', 'register_prov',
          'register_city', 'live_prov', 'live_city', 'asset_grad',
          # 'bank','bank_city', 'bank_prov', 'cardkind', 'cardtype'
          ]

applyfp = apply_all[xy_col]

applyfp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\applyfp_1108.csv', index=False)

# apply_all.head(100).to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\UnitTest\Axis_1108_2.csv', index=False)


###########1112来计算申请资料相关的变量

applyfp = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\applyfp_1108.csv', encoding='gbk')

from CcxFpABSModel.antiLogic import f_antiLogic

Antiapplyfp = f_antiLogic(applyfp)

from CcxFpABSModel.calMain import f_calMain

var = f_calMain(Antiapplyfp, ccx_Rawdata)

var.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\var_1112.csv', index=False)

var.shape
'''
(59495, 299)
'''

##########################计算出评分

from CcxFpABSModel.model import get_model_path, load_model, predict_score, model_data, score

model_path = get_model_path('model_Fp_Ccx_All_2017-11-05.txt')
bst = load_model(model_path)
pvalue = predict_score(model_data(var[bst.feature_names]), bst)
pre_score = score(pvalue)
pd.Series(pre_score).describe()
'''
count    59495.000000
mean       661.722595
std         17.690050
min        594.000000
25%        651.000000
50%        665.000000
75%        675.000000
max        696.000000
'''


def f_mdscore_(res, VAR):
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
    return rule(res, value)


mdScore = f_mdscore_(pre_score, var)
pd.Series(mdScore).describe()

'''
count    59495.000000
mean       661.669334
std         17.680928
min        527.500000
25%        651.000000
50%        665.000000
75%        675.000000
max        696.000000
'''

re = {'lend_request_id': var.lend_request_id,
      'originScore': pre_score, 'mdScore': mdScore,
      'Contract_period': var.Contract_period,
      'Contract_amount': var.Contract_amount,
      'age': var.age
      }

# var.filter(regex='Contract').columns

pd.DataFrame(re).to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\Predict_1101NewData\rule_1112.csv', index=False)
