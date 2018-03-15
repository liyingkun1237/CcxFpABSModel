###读取train test 的index
import pandas as pd

train = pd.read_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\lyk_A\rerun_model_0929\data\train_0929.csv')
test = pd.read_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\lyk_A\rerun_model_0929\data\test_0929.csv')

train_index = train[['lend_request_id', 'TargetBad_P12']]
test_index = test[['lend_request_id', 'TargetBad_P12']]

########
FPALLVAR = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\FPALLVAR_1103.csv')
train = pd.merge(train_index, FPALLVAR)
test = pd.merge(test_index, FPALLVAR)

######
train_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\train_1103.csv'  # 训练集数据路径，可见demo_data文件夹
test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\test_1103.csv'  # 测试集数据路径，可见demo_data文件夹

# train.to_csv(train_path, index=False)
# test.to_csv(test_path, index=False)
############
from ccxmodel.modelmain import ModelMain

index_name = 'lend_request_id'  # 数据集唯一索引，有且仅支持一个索引，不支持多个索引
target_name = 'TargetBad_P12'  # 目标变量
modelmain = ModelMain(train_path, test_path, index_name, target_name)

modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\ccxboost_1.conf')

r"""
{'colsample_bytree': 0.80000000000000004, 'eta': 0.10000000000000001, 'eval_metric': 'auc', 'gamma': 2, 'lambda': 500, 'max_depth': 4, 'max_test_auc': 0.71066600000000002, 'max_train_auc': 0.75396439999999987, 'min_child_weight': 2, 'num_best_round': 144, 'num_boost_round': 500, 'num_maxtest_round': 494, 'objective': 'binary:logistic', 'subsample': 0.80000000000000004, 'gap': 0.042999999999999997}

模型保存成功 文件路径名：C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171103214623/modeltxt/model_Fp_Ccx_All_2017-11-03.txt
重要变量的个数：145
数据保存成功:C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171103214623/modeldata/d_2017-11-03_importance_var.csv
训练集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92     12354
        1.0       0.77      0.02      0.04      2068
avg / total       0.85      0.86      0.80     14422
测试集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92      5295
        1.0       0.69      0.01      0.02       887
avg / total       0.83      0.86      0.79      6182
ks_train: 0.390661,ks_test：0.337822

"""

modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\ccxboost_2.conf')

r"""
{'colsample_bytree': 0.80000000000000004, 'eta': 0.10000000000000001, 'eval_metric': 'auc', 'gamma': 5, 'lambda': 500, 'max_depth': 5, 'max_test_auc': 0.70173940000000001, 'max_train_auc': 0.71659779999999995, 'min_child_weight': 2, 'num_best_round': 105, 'num_boost_round': 500, 'num_maxtest_round': 499, 'objective': 'binary:logistic', 'subsample': 0.80000000000000004, 'gap': 0.014999999999999999}

模型保存成功 文件路径名：C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171104134827/modeltxt/model_Fp_Ccx_All_2017-11-04.txt
重要变量的个数：56
数据保存成功:C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171104134827/modeldata/d_2017-11-04_importance_var.csv
训练集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92     12354
        1.0       1.00      0.00      0.00      2068
avg / total       0.88      0.86      0.79     14422
测试集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92      5295
        1.0       0.00      0.00      0.00       887
avg / total       0.73      0.86      0.79      6182

ks_train: 0.332528,ks_test：0.332283

"""

########只选重要的变量进行建模
import pandas as pd

imppath = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171104134827/modeldata/d_2017-11-04_importance_var.csv'
impvar = pd.read_csv(imppath).Feature_Name.values.tolist()

impvar_ = ['lend_request_id', 'TargetBad_P12'] + impvar

trainImp = pd.read_csv(train_path)[impvar_]
testImp = pd.read_csv(test_path)[impvar_]

trainImp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\trainImp_1104.csv', index=False)
testImp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp_1104.csv', index=False)

#####
train_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\trainImp_1104.csv'  # 训练集数据路径，可见demo_data文件夹
test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp_1104.csv'  # 测试集数据路径，可见demo_data文件夹

index_name = 'lend_request_id'  # 数据集唯一索引，有且仅支持一个索引，不支持多个索引
target_name = 'TargetBad_P12'  # 目标变量
modelmain = ModelMain(train_path, test_path, index_name, target_name)

modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\ccxboost_3.conf')

r"""
'colsample_bytree': 0.80000000000000004, 'eta': 0.10000000000000001, 'eval_metric': 'auc', 'gamma': 5, 'lambda': 300, 'max_depth': 5, 'max_test_auc': 0.70544760000000006, 'max_train_auc': 0.72430519999999998, 'min_child_weight': 2, 'num_best_round': 136, 'num_boost_round': 500, 'num_maxtest_round': 480, 'objective': 'binary:logistic', 'subsample': 0.80000000000000004, 'gap': 0.019}

模型保存成功 文件路径名：C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171104164120/modeltxt/model_Fp_Ccx_All_2017-11-04.txt
重要变量的个数：54
数据保存成功:C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171104164120/modeldata/d_2017-11-04_importance_var.csv
训练集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92     12354
        1.0       1.00      0.00      0.00      2068
avg / total       0.88      0.86      0.79     14422
测试集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92      5295
        1.0       0.00      0.00      0.00       887
avg / total       0.73      0.86      0.79      6182

ks_train: 0.343730,ks_test：0.334852

"""

####总结
# 输入变量 56个 重要变量54个
# ks_train: 0.343730,ks_test：0.334852
# auc_train:0.7267, auc_test: 0.7217


############################1105 解决新的bug后，重跑模型

train = pd.read_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\lyk_A\rerun_model_0929\data\train_0929.csv')
test = pd.read_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\lyk_A\rerun_model_0929\data\test_0929.csv')

train_index = train[['lend_request_id', 'TargetBad_P12']]
test_index = test[['lend_request_id', 'TargetBad_P12']]

'''为了联动跑模型 1107'''
train_index.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\联动优势\train_index.csv', index=False)
test_index.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\联动优势\test_index.csv', index=False)

########
FPALLVAR = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\FPALLVAR_1105.csv')
train = pd.merge(train_index, FPALLVAR)
test = pd.merge(test_index, FPALLVAR)

######
train_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\train_1105.csv'  # 训练集数据路径，可见demo_data文件夹
test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\test_1105.csv'  # 测试集数据路径，可见demo_data文件夹

# train.to_csv(train_path, index=False)
# test.to_csv(test_path, index=False)

from ccxmodel.modelmain import ModelMain

index_name = 'lend_request_id'  # 数据集唯一索引，有且仅支持一个索引，不支持多个索引
target_name = 'TargetBad_P12'  # 目标变量
modelmain = ModelMain(train_path, test_path, index_name, target_name)

modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\ccxboost_3.conf')

r"""
{'colsample_bytree': 0.80000000000000004, 'eta': 0.10000000000000001, 'eval_metric': 'auc', 'gamma': 5, 'lambda': 300, 'max_depth': 5, 'max_test_auc': 0.70542340000000003, 'max_train_auc': 0.72562040000000005, 'min_child_weight': 2, 'num_best_round': 109, 'num_boost_round': 500, 'num_maxtest_round': 410, 'objective': 'binary:logistic', 'subsample': 0.80000000000000004, 'gap': 0.02}
模型保存成功 文件路径名：C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105201306/modeltxt/model_Fp_Ccx_All_2017-11-05.txt
重要变量的个数：72
数据保存成功:C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105201306/modeldata/d_2017-11-05_importance_var.csv
训练集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92     12354
        1.0       0.80      0.00      0.00      2068
avg / total       0.85      0.86      0.79     14422
测试集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92      5295
        1.0       1.00      0.00      0.00       887
avg / total       0.88      0.86      0.79      6182
ks_train: 0.346903,ks_test：0.333924


"""

######重要变量
imppath = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105201306\modeldata\d_2017-11-05_importance_var.csv'
impvar = pd.read_csv(imppath).Feature_Name.values.tolist()

impvar_ = ['lend_request_id', 'TargetBad_P12'] + impvar

trainImp = pd.read_csv(train_path)[impvar_]
testImp = pd.read_csv(test_path)[impvar_]

trainImp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\trainImp_1105.csv', index=False)
testImp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp_1105.csv', index=False)

#####
train_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\trainImp_1105.csv'  # 训练集数据路径，可见demo_data文件夹
test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp_1105.csv'  # 测试集数据路径，可见demo_data文件夹

index_name = 'lend_request_id'  # 数据集唯一索引，有且仅支持一个索引，不支持多个索引
target_name = 'TargetBad_P12'  # 目标变量
modelmain = ModelMain(train_path, test_path, index_name, target_name)

modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\ccxboost_3.conf')

r"""
'colsample_bytree': 0.80000000000000004, 'eta': 0.10000000000000001, 'eval_metric': 'auc', 'gamma': 5, 'lambda': 300, 'max_depth': 5, 'max_test_auc': 0.70486899999999997, 'max_train_auc': 0.7245807999999998, 'min_child_weight': 2, 'num_best_round': 124, 'num_boost_round': 500, 'num_maxtest_round': 458, 'objective': 'binary:logistic', 'subsample': 0.80000000000000004, 'gap': 0.02}

重要变量的个数：60
数据保存成功:C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105210849/modeldata/d_2017-11-05_importance_var.csv
训练集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92     12354
        1.0       0.86      0.00      0.01      2068
avg / total       0.86      0.86      0.79     14422
测试集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92      5295
        1.0       1.00      0.00      0.00       887
avg / total       0.88      0.86      0.79      6182
ks_train: 0.347046,ks_test：0.333363

"""

#####再走一遍重要变量 即输入60个变量

imppath = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105210849\modeldata\d_2017-11-05_importance_var.csv'
impvar = pd.read_csv(imppath).Feature_Name.values.tolist()

impvar_ = ['lend_request_id', 'TargetBad_P12'] + impvar

trainImp = pd.read_csv(train_path)[impvar_]
testImp = pd.read_csv(test_path)[impvar_]

trainImp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\trainImp2_1105.csv', index=False)
testImp.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp2_1105.csv', index=False)

#####
train_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\trainImp2_1105.csv'  # 训练集数据路径，可见demo_data文件夹
test_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\testImp2_1105.csv'  # 测试集数据路径，可见demo_data文件夹

index_name = 'lend_request_id'  # 数据集唯一索引，有且仅支持一个索引，不支持多个索引
target_name = 'TargetBad_P12'  # 目标变量
modelmain = ModelMain(train_path, test_path, index_name, target_name)

modelmain.ccxboost_main(r'C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\ccxboost_3.conf')

r"""
{'colsample_bytree': 0.80000000000000004, 'eta': 0.10000000000000001, 'eval_metric': 'auc', 'gamma': 5, 'lambda': 300, 'max_depth': 5, 'max_test_auc': 0.70485960000000003, 'max_train_auc': 0.72449580000000002, 'min_child_weight': 2, 'num_best_round': 113, 'num_boost_round': 500, 'num_maxtest_round': 418, 'objective': 'binary:logistic', 'subsample': 0.80000000000000004, 'gap': 0.02}

[417]	test-auc:0.722863	train-auc:0.729093
模型保存成功 文件路径名：C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105211851/modeltxt/model_Fp_Ccx_All_2017-11-05.txt
重要变量的个数：57
数据保存成功:C:\Users\liyin\Desktop\CcxFpABSModel\updateModel\model20171105211851/modeldata/d_2017-11-05_importance_var.csv
训练集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92     12354
        1.0       0.75      0.00      0.00      2068
avg / total       0.84      0.86      0.79     14422
测试集模型报告：
              precision    recall  f1-score   support
        0.0       0.86      1.00      0.92      5295
        1.0       1.00      0.00      0.01       887
avg / total       0.88      0.86      0.79      6182
ks_train: 0.348392,ks_test：0.331041

"""

###############输入变量列表 60
['asset_grad_E',
 'asset_grad_D',
 'verf_is_failed',
 'verf_institution_sum',
 '_age_5.0',
 'regprov_isequal_liveprov',
 'guaduation_status_1.0',
 'verf_is_success',
 'mobile_city_level_3.0',
 'resource_type_2.0',
 'score_institution_sum',
 '_age_4.0',
 'lend_company_type_5.0',
 'overdue_cuiqian_times',
 'asset_grad_B',
 '_loan_amount_1.0',
 'gender',
 'marriage_3.0',
 'job_title_2.0',
 'idno_prov_level_2.0',
 'score_query_times',
 'bank_verf_sucess',
 'annual_income',
 'age',
 'live_prov_level_4.0',
 'MobileType_lt',
 'have_children',
 '_life_years_2.0',
 'DIR',
 'Contract_period',
 'lend_company_type_4.0',
 'have_others',
 'live_city_level_5.0',
 'job_title_4.0',
 'regcity_isequal_livecity',
 'wla_diff_years',
 'iNatives',
 '_loan_amount_3.0',
 'life_years',
 'interval_date_guaduation',
 'AMT_SUBCONAM',
 'register_prov_level_2.0',
 'verf_times',
 'work_years',
 'Borrower_monthincome',
 'educational_2.0',
 'work_unit_cate_5.0',
 'Contract_amount',
 'asset_grad_A',
 'Debt_income_ratio',
 'job_title_1.0',
 'qq_length',
 'have_friends',
 'loan_amount',
 'register_prov_level_1.0',
 'car_property_type_3.0',
 'card_is_success',
 'marriage_1.0',
 'car_property_type_1.0',
 'la_diff_years']
