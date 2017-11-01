"""
更新说明，DIR提取逻辑写在这
"""
import pandas as pd
from CcxFpABSModel.base_addr import base_addr
import numpy as np
import pickle
import os


class fp_base(base_addr):
    def __init__(self):
        # super(base_addr, self).__init__(fp_base)
        file_path = os.path.split(os.path.realpath(__file__))[0]
        varall_path = os.path.join(file_path, 'exData', 'fp_base_var.pkl')
        # varall_path = r'C:\Users\liyin\Desktop\FP_ABS\fp_base_var.pkl'

        with open(varall_path, 'rb') as f:
            self.fp_base_var = pickle.load(f)

    @staticmethod
    def f_work_uint(x):
        '''
        #行+局+院+所+司+厂+学+队+ 校+心+府+会+处+站
        :param x:
        :return:
        '''
        if x[-1] == '行':
            return 1
        elif x[-1] == '局':
            return 2
        elif x[-1] == '院':
            return 3
        elif x[-1] == '所':
            return 4
        elif x[-1] == '司':
            return 5
        elif x[-1] == '厂':
            return 6
        elif x[-1] == '学':
            return 7
        elif x[-1] == '队':
            return 8
        elif x[-1] == '校':
            return 9
        elif x[-1] == '心':
            return 10
        elif x[-1] == '府':
            return 11
        elif x[-1] == '会':
            return 12
        elif x[-1] == '处':
            return 13
        elif x[-1] == '站':
            return 14
        else:
            return np.nan

    def get_fp_base_data(self, fp_base):
        sel_col = ['lend_request_id', 'Borrower_monthincome',
                   'Contract_period', 'Contract_amount', 'asset_grad',
                   'Borrower_area_level', 'Debt_income_ratio', 'borcity_isequal_regcity',
                   'borgcity_isequal_livecity', 'city_isequal', 'live_city_level',
                   'live_prov_level', 'regcity_isequal_livecity', 'register_city_level',
                   'register_prov_level', 'regprov_isequal_liveprov', 'work_unit_cate',
                   'DIR']  # 收入负债比

        fp_var = fp_base.assign(
            Debt_income_ratio=lambda x: (x.Contract_amount / x.Contract_period) / x.Borrower_monthincome,
            # work_unit_cate=lambda x: x.Work_unit.apply(self.f_work_uint), #替换的原因为上线后凡普脱敏后即为此
            work_unit_cate=lambda x: x.Work_unit,
            register_prov_level=lambda x: x.register_prov.apply(self.f_dict_prov_level),
            register_city_level=lambda x: x.register_city.apply(self.f_dict_city_level),
            live_prov_level=lambda x: x.live_prov.apply(self.f_dict_prov_level),
            live_city_level=lambda x: x.live_city.apply(self.f_dict_city_level),
            Borrower_area_level=lambda x: x.Borrower_area.apply(self.f_dict_city_level),
            ##地区（城市）的比对 ##两个省的比对
            regprov_isequal_liveprov=lambda x: self.f_2compare_addr(x.register_prov, x.live_prov),
            regcity_isequal_livecity=lambda x: self.f_2compare_addr(x.register_city, x.live_city),
            borgcity_isequal_livecity=lambda x: self.f_2compare_addr(x.Borrower_area, x.live_city),
            borcity_isequal_regcity=lambda x: self.f_2compare_addr(x.Borrower_area, x.register_city),
            city_isequal=lambda x: self.f_3compare_addr(x.Borrower_area, x.register_city, x.live_city),
        )[sel_col]

        return fp_var

    def get_fp_base_var(self, fp_base):
        var = self.get_fp_base_data(fp_base)
        dummy_col = ['asset_grad', 'Borrower_area_level', 'live_city_level',
                     'live_prov_level', 'register_city_level',
                     'register_prov_level', 'work_unit_cate']
        var = pd.get_dummies(var, columns=dummy_col, dummy_na=True)
        return self.f_keep_dummy(dummy_col, var)[self.fp_base_var.columns]

    def f_keep_dummy(self, col, var):
        '''

        :param col: 需要dummy的列名
        :param var: dummy后的数据集
        :return: 和原训练样本保持一致数据结构的数据集
        '''
        var = pd.concat([self.fp_base_var, var]).drop('all')
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


if __name__ == '__main__':
    fp_base_rawdata = pd.read_csv(r'\\LIYINGKUN\Ccx_Fp_ABS\lyk_A\fp_base.csv', encoding='utf-8')
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

    fp_base_rawdata = fp_base_rawdata.rename(columns=dict_name)

    fp_base_preonehotdata = fp_base().get_fp_base_data(fp_base_rawdata)
    fp_base_var = fp_base().get_fp_base_var(fp_base_rawdata)

    """
    凡普模型更新，
    1）变量思路：
        借款人收入 Borrower income
        合同金额 Contract amount
        合同期限 Contract period
        
        （合同金额/合同期限）/ 借款人收入  （支出占收入的比例）Debt to income ratio
        借款人地区 Borrower area
        借款人工作单位  --》类别归约 Work unit
        户籍省份 register prov
        户籍城市 register city
        居住省份 live prov
        居住城市 live city
        资产分级 asset grad
    """

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

    fp_base = fp_base.rename(columns=dict_name)

    import re


    def f_2equal_addr(x, y):
        if x not in [None, np.nan] and y not in [None, np.nan]:
            x = str(x)
            y = str(y)
            if len(x) > len(y):
                return re.search(y, x) is not None
            elif len(x) <= len(y):
                return re.search(x, y) is not None
            else:
                return False
        else:
            return False


    def f_3compare_addr(x, y, z):
        ls = []
        for i, j, k in zip(x, y, z):
            if f_2equal_addr(i, j) and f_2equal_addr(j, k):
                ls.append(1)
            else:
                ls.append(0)
        return ls


    def f_2compare_addr(x, y):
        ls = []
        for i, j in zip(x, y):
            if f_2equal_addr(i, j):
                ls.append(1)
            else:
                ls.append(0)
        return ls


    def f_work_uint(x):
        '''
        #行+局+院+所+司+厂+学+队+ 校+心+府+会+处+站
        :param x:
        :return:
        '''
        if x[-1] == '行':
            return 1
        elif x[-1] == '局':
            return 2
        elif x[-1] == '院':
            return 3
        elif x[-1] == '所':
            return 4
        elif x[-1] == '司':
            return 5
        elif x[-1] == '厂':
            return 6
        elif x[-1] == '学':
            return 7
        elif x[-1] == '队':
            return 8
        elif x[-1] == '校':
            return 9
        elif x[-1] == '心':
            return 10
        elif x[-1] == '府':
            return 11
        elif x[-1] == '会':
            return 12
        elif x[-1] == '处':
            return 13
        elif x[-1] == '站':
            return 14
        else:
            return np.nan


    sel_col = ['lend_request_id', 'Borrower_monthincome',
               'Contract_period', 'Contract_amount', 'asset_grad',
               'Borrower_area_level', 'Debt_income_ratio', 'borcity_isequal_regcity',
               'borgcity_isequal_livecity', 'city_isequal', 'live_city_level',
               'live_prov_level', 'regcity_isequal_livecity', 'register_city_level',
               'register_prov_level', 'regprov_isequal_liveprov', 'work_unit_cate']

    fp_var = fp_base.assign(
        Debt_income_ratio=lambda x: (x.Contract_amount / x.Contract_period) / x.Borrower_monthincome,
        work_unit_cate=lambda x: x.Work_unit.apply(f_work_uint),
        register_prov_level=lambda x: x.register_prov.apply(base_addr.f_dict_prov_level),
        register_city_level=lambda x: x.register_city.apply(base_addr.f_dict_city_level),
        live_prov_level=lambda x: x.live_prov.apply(base_addr.f_dict_prov_level),
        live_city_level=lambda x: x.live_city.apply(base_addr.f_dict_city_level),
        Borrower_area_level=lambda x: x.Borrower_area.apply(base_addr.f_dict_city_level),
        ##地区（城市）的比对 ##两个省的比对
        regprov_isequal_liveprov=lambda x: f_2compare_addr(x.register_prov, x.live_prov),
        regcity_isequal_livecity=lambda x: f_2compare_addr(x.register_city, x.live_city),
        borgcity_isequal_livecity=lambda x: f_2compare_addr(x.Borrower_area, x.live_city),
        borcity_isequal_regcity=lambda x: f_2compare_addr(x.Borrower_area, x.register_city),
        city_isequal=lambda x: f_3compare_addr(x.Borrower_area, x.register_city, x.live_city),
    )[sel_col]

    dummy_col = ['asset_grad', 'Borrower_area_level', 'live_city_level',
                 'live_prov_level', 'register_city_level',
                 'register_prov_level', 'work_unit_cate']

    fp_base_var = pd.get_dummies(fp_var, columns=dummy_col, dummy_na=True)
    fp_base_var.to_csv(r'C:\Users\liyin\Desktop\fp_model_ABS\fp_base_var_0920.csv', index=False)

    fp_base_var.columns

    fp_var.to_csv(r'C:\Users\liyin\Desktop\fp_model_ABS\fp_basevarpreonehot_0920_.csv', index=False)

    ###################0925 对抽样的数据进行计算变量 ############
    fp_base = pd.read_excel(r'\\10.0.31.245\Ccx_Fp_ABS\Data\0920_data\中诚信需求数据_0920\2）随机抽样客户字段表.xlsx',
                            sheetindex=0)
    fp_base_var.rename(columns=dict_name, inplace=True)

    fp_base_var.to_csv(r'\\10.0.31.245\Ccx_Fp_ABS\lyk_A\data_0925\fp_base_var.csv', index=False)

    fp_base_var.columns

    ###############
    x = fp_base_var.head(1)
    x.index = ['all']
    x

    with open(r'C:\Users\liyin\Desktop\FP_ABS\fp_base_var.pkl', 'wb') as f:
        pickle.dump(x, f)

    with open(r'C:\Users\liyin\Desktop\FP_ABS\fp_base_var.pkl', 'rb') as f:
        x = pickle.load(f)

    #####20171030  加入DIR 收入负债比数据
    x['DIR'] = 0.01
    x

    with open(r'C:\Users\liyin\Desktop\FP_ABS\fp_base_var.pkl', 'wb') as f:
        pickle.dump(x, f)

    with open(r'C:\Users\liyin\Desktop\FP_ABS\fp_base_var.pkl', 'rb') as f:
        x = pickle.load(f)
