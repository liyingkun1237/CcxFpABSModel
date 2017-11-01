import pickle
import numpy as np
import pandas as pd
import os


class fp_apply(object):
    '''
    依据凡普的申请资料数据，对数据进行数据清洗，特征归约，变量衍生，变量编码等操作 最终返回建模用数据

    '''

    def __init__(self):
        '''
        初始化函数设计思路，接收fp数据更改为传入全样本数据计算出的列，而将凡普数据作为参赛传递给
        返回变量的计算函数
        优点：少些一些self
        :param fp:
        '''
        file_path = os.path.split(os.path.realpath(__file__))[0]
        fp_allvar_path = os.path.join(file_path, 'exData', 'fp_varall.pkl')
        # fp_allvar_path = r'C:\Users\liyin\Desktop\FP_ABS\fp_varall.pkl'
        with open(fp_allvar_path, 'rb') as f:
            self.fp_varall = pickle.load(f)

    def get_fp_apply_data(self, fpdata):
        '''
        计算出凡普的one - hot 之前的数据
        :param fpdata:原始的数据
        :return:
        '''

        fp_apply_data = fpdata.assign(
            # 类别转换 类别归约
            # loan_purpose=lambda x: x.loan_purpose.apply(self.f_dict_loan_purpose),
            # gender=lambda x: x.gender.apply(self.f_dict_gender),
            # education=lambda x: x.education.apply(self.f_dict_education),
            # marriage=lambda x: x.marriage.apply(self.f_dict_marriage),
            # iNatives=lambda x: x.iNatives.apply(self.f_dict_bio),
            # house_property_type=lambda x: x.house_property_type.apply(self.f_dict_house_property),
            # resource_type=lambda x: x.resource_type.apply(self.f_dict_resource_type),
            # lend_company_type=lambda x: x.lend_company_type.apply(self.f_dict_lend_company),
            # job_title=lambda x: x.job_title.apply(self.f_dict_job_title),
            # department=lambda x: x.department.apply(self.f_dict_department),
            # car_property_type=lambda x: x.car_property_type.apply(self.f_dict_car_property),
            # secret_to_family=lambda x: x.secret_to_family.apply(self.f_dict_bio),
            # 注释掉的原因，凡普数据脱敏后直接为这样了
            # 分箱操作
            _loan_amount=lambda x: x.loan_amount.apply(self.f_dict_loan_amount),
            _age=lambda x: x.age.apply(self.f_dict_age),
            _annual_income=lambda x: x.annual_income.apply(self.f_dict_annual_income),
            _life_years=lambda x: x.life_years.apply(self.f_dict_life_years),
            _work_years=lambda x: x.work_years.apply(self.f_dict_work_years),
            # 衍生变量 life_year - work_years  life_years-age
            wla_diff_years=lambda x: x.life_years - x.work_years,
            la_diff_years=lambda x: x.life_years - x.age,
        )

        return fp_apply_data

    def get_fp_apply_var(self, fpdata):
        '''
        从原始数据，计算出凡普建模所需的模型字段
        :param fpdata:
        :return:
        '''
        col = ['loan_purpose', 'education', 'marriage', 'department',
               'house_property_type', 'resource_type', 'lend_company_type',
               'job_title', 'car_property_type', '_loan_amount', '_age',
               '_annual_income', '_life_years', '_work_years']
        var = pd.get_dummies(self.get_fp_apply_data(fpdata), columns=col, dummy_na=True)
        return self.f_keep_fpdummy(col, var)[self.fp_varall.columns]

    def f_keep_fpdummy(self, col, var):
        '''

        :param col: 需要dummy的列名
        :param var: dummy后的数据集
        :return: 和原训练样本保持一致数据结构的数据集
        '''
        var = pd.concat([self.fp_varall, var]).drop('all')
        # 找出one-hot 之后的列
        ls = []
        for i in col:
            ls += list(var.filter(regex=i).columns.values)
        fill_dict = {}
        for x in ls:
            fill_dict[x] = 0
        # print(fill_dict)
        var = var.fillna(fill_dict)
        return var

    @staticmethod
    def f_dict_bio(x):
        '''
        # iNatives  secret_to_family
        二分类数据 映射
        0 - NO 否
        1 - YES 是
        '''
        if x == 'NO' or x == '否':
            return 0
        elif x == 'YES' or x == '是':
            return 1
        else:
            return np.nan

    @staticmethod
    def f_dict_gender(x):
        '''
        # gender
        :param x:
        :return:
        '''
        if x == 'FEMALE':
            return 0
        elif x == 'MALE':
            return 1
        else:
            return np.nan

    @staticmethod
    def f_dict_education(x):
        '''
        # education
        :return:
        '''
        dict_education = {'ZHUANKE': 1,  # 专科
                          'UNDERGRADUATE': 2,  # 本科
                          'SEN': 3,  # 高中
                          'MASTER': 4}  # 研究生
        if str(x).strip() in dict_education.keys():
            return dict_education[x]
        else:
            return np.nan

    @staticmethod
    def f_dict_marriage(x):
        '''
        # marriage
        :return:
        '''
        dict_marriage = {
            'MARRIED': 1,
            'UNMARRIED': 2,
            'DIVORCED': 3,
            'WIDOWED': 4,
        }

        if x in dict_marriage.keys():
            return dict_marriage[x]
        else:
            return np.nan

    @staticmethod
    def f_dict_house_property(x):
        '''
        # house_property_type
        :return:
        '''
        dict_house_property = {'HAVE_HOUSE_AND_LOAN_WITHOUT': 1,
                               'HAVE_HOUSE_AND_LOAN': 2,
                               'NO_HOUSE': 3}
        if x in dict_house_property.keys():
            return dict_house_property[x]
        else:
            return np.nan

    @staticmethod
    def f_dict_resource_type(x):
        '''
        # resource_type
        :return:
        OTHERS              7471
        OUT_SEND            1887
        TELEPHONE_SALE      1350
        FRIEND               263

        ADS
        WEBSITE               38
        MEDIA                 12
        SHORT_MESSAGE          3
        MARKTINGACTIVITY       1
        '''
        dict_resource_type = {'OTHERS': 1,
                              'OUT_SEND': 2,
                              'TELEPHONE_SALE': 3,
                              'FRIEND': 4,

                              'WEBSITE': 5,
                              'MEDIA': 5,
                              'SHORT_MESSAGE': 5,
                              'MARKTINGACTIVITY': 5}
        if x in dict_resource_type.keys():
            return dict_resource_type[x]
        else:
            return np.nan

    @staticmethod
    def f_dict_lend_company(x):
        '''
        # lend_company_type
        :return:
        '''
        dict_lend_company = {'ENTERPRISE_COMPANY': 1,
                             'LIST_COMPANY': 2,
                             'OPERATE_COMPANY': 3,
                             'JOINT_VENTURE': 4,
                             'OUT_COMPANY': 5,
                             'PRIVATE_COMPANY': 6,
                             'OTHER_COMPANY ': 7}
        if x in dict_lend_company.keys():
            return dict_lend_company[x]
        else:
            return np.nan

    @staticmethod
    def f_dict_loan_purpose(x):
        '''
        # loan_purpose
        :return:
        '''
        purpose_dict = {'副业经营': 'sideline',
                        '装修': 'renovation',
                        '购车': 'buycar',
                        '购房': 'buyhouse',
                        '家电/数码产品消费 ': 'digital',
                        '其他': 'other',
                        '资金周转': 'capitalturnover',
                        '教育培训': 'education'}

        # {'副业经营': 'sideline',:1,
        #  '装修': 'renovation',:2,
        #  '购车': 'buycar',:3,
        #  '购房': 'buyhouse',:4,
        #  '家电/数码产品消费 ': 'digital',:5,
        #  '其他': 'other',:6,
        #  '资金周转': 'capitalturnover':7,
        #  '教育培训': 'education':8}

        if x in purpose_dict.keys():
            return purpose_dict[x]
        elif x not in purpose_dict.keys():
            return purpose_dict['其他']
        else:
            return np.nan

    @staticmethod
    def f_dict_department(x):
        '''
        # 处理思路2：部门归类
        :return:
        部门较为分散，4977个
        处理思路1：直接作为连续变量处理
        处理思路2：部门归类  组 部 科 所  室
        '''

        if x[-1] == '部':
            return 1
        elif x[-1] == '科':
            return 2
        elif x[-1] == '所':
            return 3
        elif x[-1] == '处':
            return 4
        elif x[-1] == '室':
            return 5
        elif x[-1] == '办':
            return 6
        elif x[-1] == '组':
            return 7
        elif x[-1] == '局':
            return 8
        elif x[-1] == '队':
            return 9
        elif x[-2:] == '中心':
            return 10
        else:
            return 11

    @staticmethod
    def f_dict_job_title(x):
        '''
        # job_title
        :return:
        '''
        dict_job = {'一般正式员工': 1,
                    '一般管理人员': 2,
                    '中级管理人员': 3,
                    '高级管理人员': 4,
                    '退休人员': 5,
                    '负责人': 6,
                    '派遣员工': 7,
                    '其他': 8}
        if x in dict_job.keys():
            return dict_job[x]
        elif x not in dict_job.keys():
            return dict_job['其他']
        else:
            return np.nan

    @staticmethod
    def f_dict_car_property(x):
        '''
        # car_property_type
        :return:
        '''
        dict_car_property = {
            'NO_CAR': 1,
            'HAVE_CAR_AND_LOAN_WITHOUT': 2,
            'HAVE_CAR_AND_LOAN': 3}

        if x in dict_car_property.keys():
            return dict_car_property[x]
        else:
            return np.nan

    @staticmethod
    def f_dict_annual_income(x):
        '''
        对连续型变量 进行分箱 并one - hot 转换 ， 与连续变量归一化
        # ['loan_amount', 'age','annual_income', 'life_years','work_years']

         # annual_income
        :return:
        '''
        if 0 <= x < 60000:
            return 1
        elif 60000 <= x < 100000:
            return 2
        elif 100000 <= x < 200000:
            return 3
        elif 200000 <= x < 400000:
            return 4
        elif 400000 <= x < 1000000:
            return 5
        elif 1000000 <= x < np.Inf:
            return 6
        else:
            return np.nan

    @staticmethod
    def f_dict_loan_amount(x):
        '''
        # loan_amount
        :return:
        '''
        if 0 <= x < 60000:
            return 1
        elif 60000 <= x < 80000:
            return 2
        elif 80000 <= x < 100000:
            return 3
        elif 100000 <= x < 120000:
            return 4
        elif 120000 <= x < 150000:
            return 5
        elif 150000 <= x < np.Inf:
            return 6
        else:
            return np.nan

    @staticmethod
    def f_dict_work_years(x):
        '''
        # work_years
        :return:
        '''
        if 0 <= x < 10:
            return 1
        elif 10 <= x < 18:
            return 2
        elif 18 <= x < 25:
            return 3
        elif 25 <= x < 35:
            return 4
        elif 35 <= x < 46:
            return 5
        else:
            return np.nan

    @staticmethod
    def f_dict_life_years(x):
        '''
        # life_years
        :return:
        '''
        if 0 <= x < 20:
            return 1
        elif 20 <= x < 28:
            return 2
        elif 28 <= x < 35:
            return 3
        elif 35 <= x < 45:
            return 4
        elif 45 <= x < 61:
            return 5
        else:
            return np.nan

    @staticmethod
    def f_dict_age(x):
        '''
        # age
        :return:
        '''
        if 22 <= x < 28:
            return 1
        elif 28 <= x < 32:
            return 2
        elif 32 <= x < 35:
            return 3
        elif 35 <= x < 45:
            return 4
        elif 45 <= x < 61:
            return 5
        else:
            return np.nan


if __name__ == '__main__':
    # apply_data_all = pd.read_table(r'C:\Users\liyin\Desktop\fp_model_ABS\apply_data_all.txt', sep='\t', encoding='gbk')
    #
    # fp_preonehot_data = fp_apply().get_fp_apply_data(apply_data_all)
    # fp_postonehot_data = fp_apply().get_fp_apply_var(apply_data_all)
    # with open(r'C:\Users\liyin\Desktop\FP_ABS\varall.pkl', 'wb') as f:
    #     pickle.dump(fp_all, f)
    # fp_preonehot_data.to_csv(r'C:\Users\liyin\Desktop\fp_model_ABS\fp_preonehot_data.csv', index=False)
    # fp_postonehot_data.to_csv(r'C:\Users\liyin\Desktop\fp_model_ABS\fp_apply_var_0920.csv', index=False)
    # fp_preonehot_data.to_csv(r'\\10.0.31.245\Ccx_Fp_ABS\lyk_A\data_0921\fp_apply_preonehot.csv', index=False)

    ''' 凡普第三批数据 0925'''
    dict_name = {
        '进件号': 'lend_request_id',
        '借款人收入': 'Borrower_monthincome',
        '合同金额': 'Contract_amount',
        '合同期限': 'Contract_period',
        '借款人地区': 'Borrower_area',
        '借款人工作单位': 'Work_unit',
        '户籍省份': 'register_prov',
        '户籍城市': 'register_city',
        '居住省份': 'live_prov',
        '居住城市': 'live_city',
        '资产分级': 'asset_grad'
    }
    apply_data_all = pd.read_excel(r'\\10.0.31.245\Ccx_Fp_ABS\Data\0920_data\中诚信需求数据_0920\2）随机抽样客户字段表.xlsx',
                                   sheetindex=0)
    apply_data_all.rename(columns=dict_name, inplace=True)
    # 'annual_income'
    apply_data_all['annual_income'] = apply_data_all['Borrower_monthincome'] * 12
    fp_postonehot_data = fp_apply().get_fp_apply_var(apply_data_all)

    fp_postonehot_data.to_csv(r'\\10.0.31.245\Ccx_Fp_ABS\lyk_A\data_0925\fp_apply_var.csv', index=False)

    rule_0922 = pd.read_csv(r'C:\Users\liyin\Desktop\Ccx_Fp_ABS\Data\fp_rule_0922.csv')
    zy = pd.merge(rule_0922, apply_data_all)
    zy = pd.merge(zy, fp_base)

    zy.to_excel(r'\\10.0.31.245\Ccx_Fp_ABS\Data\zy_0926——3.xlsx')
    ###
