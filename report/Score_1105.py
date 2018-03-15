"""
计算 更新模型后的score

"""

import pandas as pd
import numpy as np

from CcxFpABSModel.model import score

testpred = pd.read_csv(
    r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171104164120\modeldata\d_2017-11-04_test_predict.csv')

testpred.P_value.describe()
'''
count    6182.000000
mean        0.143772
std         0.085323
min         0.037061
25%         0.080355
50%         0.115703
75%         0.181435
max         0.495562

'''

score(testpred.P_value).describe()
"""
count    6182.000000
mean      655.754934
std        18.629763
min       601.000000
25%       643.000000
50%       659.000000
75%       670.000000
max       694.000000

"""

testpred['Ccx_score'] = score(testpred.P_value)

test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp_1104.csv'  # 测试集数据路径，可见demo_data文件夹
t_ = pd.read_csv(test_path)[['lend_request_id', 'TargetBad_P12']]
ss_ = pd.concat([testpred, t_], axis=1)
ss_ = ss_.rename(columns={"TargetBad_P12": 'target'})


# ss_.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fptest_score_1105.csv', index=False)


def IV(x, y):
    crtab = pd.crosstab(x, y, margins=True)
    crtab.columns = ['good', 'bad', 'total']
    crtab['factor_per'] = crtab['total'] / len(y)
    crtab['bad_per'] = crtab['bad'] / crtab['total']
    crtab['p'] = crtab['bad'] / crtab.ix['All', 'bad']
    crtab['q'] = crtab['good'] / crtab.ix['All', 'good']
    crtab['woe'] = np.log(crtab['p'] / crtab['q'])
    iv = (crtab['p'] - crtab['q']) * np.log(crtab['p'] / crtab['q'])
    crtab['IV'] = sum(iv[(iv != np.inf) & (iv != -np.inf)])
    return crtab


def report_iv(ss_):
    ss_['bins_score'] = pd.cut(ss_.Ccx_score, np.arange(590, 701, 10))
    iv = IV(ss_['bins_score'], ss_['target'])
    iv.loc[0:len(iv) - 1, 'model_pvalue'] = ss_.groupby('bins_score')['P_value'].mean()
    # iv.loc[0:len(iv)-1, 'model_pvalue'] 这个坑好大
    return iv


report_iv(ss_).to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fpabs_testscore.csv')

# pd.__version__

############
import seaborn as sns

sns.kdeplot(ss_.Ccx_score, shade=True)
sns.distplot(ss_.Ccx_score)

#######评分修正
test_path_ = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\test_1103.csv'
t_ = pd.read_csv(test_path_)[['lend_request_id', 'TargetBad_P12', 'shixing_times', 'zhixing_vague_times']]
ss_ = pd.concat([testpred, t_], axis=1)
ss_ = ss_.rename(columns={"TargetBad_P12": 'target'})

ss_.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fptest_Mdscore_1105.csv', index=False)

#############################1105 重新计算评分

testpred = pd.read_csv(
    r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105211851\modeldata\d_2017-11-05_test_predict.csv')

testpred.P_value.describe()
r'''
count    6182.000000
mean        0.143759
std         0.086073
min         0.037764
25%         0.080440
50%         0.115372
75%         0.181243
max         0.507589
Name: P_value, dtype: float64

'''
score(testpred.P_value).describe()

'''
count    6182.000000
mean      655.812358
std        18.759888
min       599.000000
25%       643.250000
50%       659.000000
75%       670.000000
max       693.000000

'''

testpred['Ccx_score'] = score(testpred.P_value)

test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp_1104.csv'  # 测试集数据路径，可见demo_data文件夹
t_ = pd.read_csv(test_path)[['lend_request_id', 'TargetBad_P12']]
ss_ = pd.concat([testpred, t_], axis=1)
ss_ = ss_.rename(columns={"TargetBad_P12": 'target'})

# ss_.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fptest_score2_1105.csv', index=False)

report_iv(ss_).to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fpabs_testscore_1.csv')

#######使用风险信息进行评分修正
test_path_ = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\test_1105.csv'
t_ = pd.read_csv(test_path_)[['lend_request_id', 'TargetBad_P12', 'shixing_times', 'zhixing_exact_times']]
ss_ = pd.concat([testpred, t_], axis=1)
ss_ = ss_.rename(columns={"TargetBad_P12": 'target'})
ss_['sumrisk'] = ss_.shixing_times.fillna(0) + ss_.zhixing_exact_times.fillna(0)


# ss_['sumrisk'].value_counts(dropna=False)

def rule(presocre, value):
    ls = []
    for i, j in zip(presocre, value):
        if i > 670 and j > 0:  # 满足入池条件且有风险
            ls.append(i - (i - 670) * j * 1.5)
        else:
            ls.append(i)
    return ls


ss_['mdScore'] = rule(ss_.Ccx_score, ss_.sumrisk)

ss_['mdScore'].describe()
ss_['Ccx_score'].describe()


def report_iv(ss_):
    ss_['bins_score'] = pd.cut(ss_.mdScore, np.arange(590, 701, 10))
    iv = IV(ss_['bins_score'], ss_['target'])
    iv.loc[0:len(iv) - 1, 'model_pvalue'] = ss_.groupby('bins_score')['P_value'].mean()
    # iv.loc[0:len(iv)-1, 'model_pvalue'] 这个坑好大
    return iv


report_iv(ss_).to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fpabs_testscore1107_2.csv')

######评分调整规则
"""
害群之马：入池的target==1 ，规则：失信+执行次数大于等于1
1515件入池	66个害群之马	依据规则，能剔除4个	还剩62颗老鼠屎		
依据规则将误判65件，误判定义，依据规则把表现良好的人给剔除了
原因：失信越多，表现越好
"""


# rule(ss_.Ccx_score, ss_.shixing_times)


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

    value = list(VAR.shixing_times + VAR.zhixing_exact_times)
    return rule([int(res['Ccx_score'])], value)


# ss_.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\fptest_Mdscore_1106.csv', index=False)


#########################1113 计算更新模型后的评分，供模型融合使用######
import pandas as pd
path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\FPALLVAR_1105.csv'
var = pd.read_csv(path)

from CcxFpABSModel.model import get_model_path, load_model, predict_score, model_data, score

model_path = get_model_path('model_Fp_Ccx_All_2017-11-05.txt')
bst = load_model(model_path)
pvalue = predict_score(model_data(var[bst.feature_names]), bst)
pre_score = score(pvalue)
pd.Series(pre_score).describe()
'''
count    35601.000000
mean       656.601074
std         18.496820
min        596.000000
25%        644.000000
50%        660.000000
75%        671.000000
max        694.000000

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
count    35601.000000
mean       656.487051
std         18.456984
min        526.000000
25%        644.000000
50%        660.000000
75%        670.000000
max        694.000000

'''

re = {'lend_request_id': var.lend_request_id,
      'originScore': pre_score, 'mdScore': mdScore,
      'Contract_period': var.Contract_period,
      'Contract_amount': var.Contract_amount,
      'age': var.age
      }

# var.filter(regex='Contract').columns

pd.DataFrame(re).to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\report\rulePart1_1113.csv', index=False)
