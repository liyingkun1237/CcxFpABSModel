import pickle
import numpy as np
import pandas as pd
import re
import os


class base_addr(object):
    '''
    依据给定的四要素信息，计算出地址相关的变量
    '''

    def __init__(self, base):
        # idno_path = r'C:\Users\liyin\Desktop\FP_ABS\idno_dict.pkl'
        # mobile_path = r'C:\Users\liyin\Desktop\FP_ABS\mobile_addr_dict.pkl'
        # bank_path = r'C:\Users\liyin\Desktop\FP_ABS\bank_addr_dict_all.pkl'
        # # bank_path = r'C:\Users\liyin\Desktop\FP_ABS\bank_addr_dict_3rd.pkl'  # 0926更新
        # varall_path = r'C:\Users\liyin\Desktop\FP_ABS\varall.pkl'
        file_path = os.path.split(os.path.realpath(__file__))[0]

        idno_path = os.path.join(file_path, 'exData', 'idno_dict.pkl')
        mobile_path = os.path.join(file_path, 'exData', 'mobile_addr_dict.pkl')
        bank_path = os.path.join(file_path, 'exData', 'bankAddr_dict_One.pkl')
        varall_path = os.path.join(file_path, 'exData', 'varall.pkl')

        with open(idno_path, 'rb') as f:
            self.province = pickle.load(f)
            self.city = pickle.load(f)
        with open(mobile_path, 'rb') as f:
            self.mobile_addr = pickle.load(f)
        with open(bank_path, 'rb') as f:
            self.bankcard_dict = pickle.load(f)
        with open(varall_path, 'rb') as f:
            self.baseaddr_varall = pickle.load(f)

        self.base = base

    # 1.依据身份证，获取地址信息
    def get_idno_prov(self, x):
        try:
            return self.province[str(x).strip()[:2]]
        except:
            return np.nan

    def get_idno_city(self, x):
        try:
            return self.city[str(x).strip()[:4]]
        except:
            return np.nan


            # 2.依据手机号前7位，获取归属地信息

    def get_mobile_7(self, x):
        return str(x).strip()[:7]

    def get_base_addrdata(self):
        # 数据匹配
        self.base['MobileNumber'] = self.base.mobile.apply(self.get_mobile_7)
        # base_addr = pd.merge(pd.merge(self.base, self.mobile_addr, how='left', on='MobileNumber').assign(
        #     idno_prov=lambda x: x.id_no.apply(self.get_idno_prov),
        #     idno_city=lambda x: x.id_no.apply(self.get_idno_city)
        # ), self.bankcard_dict)

        base_addr_1 = pd.merge(self.base, self.mobile_addr, how='left', on='MobileNumber').assign(
            idno_prov=lambda x: x.id_no.apply(self.get_idno_prov),
            idno_city=lambda x: x.id_no.apply(self.get_idno_city)
        )

        base_addr = pd.merge(base_addr_1, self.bankcard_dict, how='left', on=['lend_request_id', 'card_no_pri'])
        base_addr = base_addr.assign(
            # 数据清洗
            bank=lambda x: x.bank.apply(self.f_bank_clean),
            MobileType=lambda x: x.MobileType.apply(self.f_mobiletype_clean),
            # 数据转换
            mobile_prov_level=lambda x: x.MobileProv.apply(self.f_dict_prov_level),
            idno_prov_level=lambda x: x.idno_prov.apply(self.f_dict_prov_level),
            bank_prov_level=lambda x: x.bank_prov.apply(self.f_dict_prov_level),

            mobile_city_level=lambda x: x.MobileCity.apply(self.f_dict_city_level),
            idno_city_level=lambda x: x.idno_city.apply(self.f_dict_city_level),
            bank_city_level=lambda x: x.bank_city.apply(self.f_dict_city_level),

            # 变量衍生
            # 地址间的比对，三者相同,记为1，否则为0. 1个变量*2个地址
            prov_is_equal=lambda x: self.f_3compare_addr(x.MobileProv, x.idno_prov, x.bank_prov),
            city_is_equal=lambda x: self.f_3compare_addr(x.MobileCity, x.idno_city, x.bank_city),
            # 地址间的两两的比较，相同为1，否则为0. 3个变量*2个地址
            mprov_isequal_idprov=lambda x: self.f_2compare_addr(x.MobileProv, x.idno_prov),
            idprov_isequal_bankprov=lambda x: self.f_2compare_addr(x.idno_prov, x.bank_prov),
            mprov_isequal_bankprov=lambda x: self.f_2compare_addr(x.MobileProv, x.bank_prov),

            mcity_isequal_idcity=lambda x: self.f_2compare_addr(x.MobileCity, x.idno_city),
            idcity_isequal_bankcity=lambda x: self.f_2compare_addr(x.idno_city, x.bank_city),
            mcity_isequal_bankcity=lambda x: self.f_2compare_addr(x.MobileCity, x.bank_city),

        ).assign(
            # 开户卡银行是否是4大行
            bigbank=lambda x: x.bank.apply(self.f_4bank),
            # 四大行一个行作为一个类目，其余作为一个类目
            bank_level=lambda x: x.bank.apply(self.f_bank_level),
        )

        return base_addr

    def f_bank_clean(self, x):
        if x not in [None, np.nan]:
            x = str(x)
            if re.search('分行', x) is not None:
                return x.strip().split('银行')[0] + '银行'
            else:
                return x.strip()
        else:
            return x

    def f_mobiletype_clean(self, x):
        if x not in [None, np.nan]:
            if re.search('移动', x) is not None:
                return 'yd'  # '移动'
            elif re.search('联通', x) is not None:
                return 'lt'  # '联通'
            elif re.search('电信', x) is not None:
                return 'dx'  # '电信'
        else:
            return x

    @staticmethod
    def f_dict_prov_level(x):
        pattern_1 = '上海|北京|天津|广东|澳门|香港'
        pattern_2 = '江苏|山东|福建|浙江|安徽|重庆'
        pattern_3 = '湖北|湖南|河南|河北|江西|山西|陕西'
        pattern_4 = '广西|海南|云南|四川|贵州'
        pattern_5 = '黑龙江|吉林|辽宁|内蒙古|甘肃|新疆|青海|宁夏|西藏'
        if re.search(pattern_1, str(x)) is not None:
            return 1
        elif re.search(pattern_2, str(x)) is not None:
            return 2
        elif re.search(pattern_3, str(x)) is not None:
            return 3
        elif re.search(pattern_4, str(x)) is not None:
            return 4
        elif re.search(pattern_5, str(x)) is not None:
            return 5
        else:
            return 6

    @staticmethod
    def f_dict_city_level(x):
        pattern_1 = '北京|上海|广州|深圳|天津'  # 一线
        pattern_2 = '杭州|南京|济南|重庆|青岛|大连|宁波|厦门'  # 二线发达
        pattern_3 = '成都|武汉|哈尔滨|沈阳|西安|长春|长沙|福州|郑州|石家庄|苏州|佛山|东莞|无锡|烟台|太原'  # 二线中等发达
        pattern_4 = '合肥|南昌|南宁|昆明|温州|淄博|唐山'  # 二线发展较弱

        if x not in [None, np.nan]:
            if re.search(pattern_1, str(x)) is not None:
                return 1
            elif re.search(pattern_2, str(x)) is not None:
                return 2
            elif re.search(pattern_3, str(x)) is not None:
                return 3
            elif re.search(pattern_4, str(x)) is not None:
                return 4
            else:
                return 5  # 三线以下城市
        else:
            return 6  # 缺失

    @staticmethod
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

    def f_3compare_addr(self, x, y, z):
        ls = []
        for i, j, k in zip(x, y, z):
            if self.f_2equal_addr(i, j) and self.f_2equal_addr(j, k):
                ls.append(1)
            else:
                ls.append(0)
        return ls

    def f_2compare_addr(self, x, y):
        ls = []
        for i, j in zip(x, y):
            if self.f_2equal_addr(i, j):
                ls.append(1)
            else:
                ls.append(0)
        return ls

    def f_4bank(self, x):
        pattern_1 = '中国银行|中国农业银行|中国建设银行|中国工商银行'
        if x not in [None, np.nan]:
            if re.search(pattern_1, str(x)) is not None:
                return 1
            else:
                return 0
        else:
            return x

    def f_bank_level(self, x):
        if x not in [None, np.nan]:
            if re.search('中国银行', str(x)) is not None:
                return 1
            elif re.search('中国农业银行', str(x)) is not None:
                return 2
            elif re.search('中国建设银行', str(x)) is not None:
                return 3
            elif re.search('中国工商银行', str(x)) is not None:
                return 4
            else:
                return 5
        else:
            return x

    def f_keep_dummy(self, var):
        '''
        self.allvar 为全样本dummy后的全部数据集 ，此函数的作用为保持数据每一个进件编码后变量一样多
        :param var: dummy后的变量
        :return:
        '''
        return pd.concat([self.baseaddr_varall, var]).drop('all').fillna(0)

    def get_base_var(self):
        col = ['bank_city_level', 'bank_prov_level', 'idno_city_level',
               'idno_prov_level', 'mobile_city_level', 'mobile_prov_level',
               'bank_level', 'MobileType']
        var = pd.get_dummies(self.get_base_addrdata(), columns=col, dummy_na=True)
        unselect_col = ['PassMth', 'name', 'mobile', 'id_no', 'card_no_pri',
                        'MobileNumber', 'MobileArea', 'MobileType', 'MobileCity', 'MobileProv',
                        'idno_city', 'idno_prov', 'bank', 'cardkind', 'cardtype', 'bank_prov',
                        'bank_city', 'bank_city_level', 'bank_prov_level', 'idno_city_level', 'idno_prov_level',
                        'mobile_city_level', 'mobile_prov_level',
                        'bank_level']
        sel_col = [x for x in var.columns if x not in unselect_col]
        var = var[sel_col]
        # print(len(sel_col))

        # 保证计算变量的变量一致性
        var = self.f_keep_dummy(var)[self.baseaddr_varall.columns]
        return var


if __name__ == '__main__':
    '''第二批凡普数据 '''
    # base = pd.read_table(r'C:\Users\liyin\Desktop\FP_ABS\fp_base_2.txt', encoding='gbk')
    # xx = base_addr(base).get_base_addrdata()
    # yy = base_addr(base).get_base_var()

    # xx.to_csv(r'C:\Users\liyin\Desktop\fp_model_ABS\base_addr_preonhot.csv', index=False)

    # dd = base_addr(base.head(1)).get_base_addrdata()
    # tt = base_addr(base.head(1)).get_base_var()
    # yy.head(1)
    # pd.concat([pd.DataFrame(yy.head(1), index=['all']), tt]).drop('all').fillna(0)

    # with open(r'C:\Users\liyin\Desktop\FP_ABS\varall.pkl', 'wb') as f:
    #     pickle.dump(pd.DataFrame(yy.head(1), index=['all']), f)

    # yy.to_csv(r'C:\Users\liyin\Desktop\fp_model_ABS\base_addr_var_0920.csv', index=False)
    # xx.to_csv(r'\\10.0.31.245\Ccx_Fp_ABS\lyk_A\data_0921\base_addr_varpreonehot.csv', index=False)

    ''' 凡普第三批数据 0925'''
    base = pd.read_excel(r'C:\Users\liyin\Desktop\FP_ABS\fp_base_3.xlsx', encoding='gbk')
    base_dict = {'进件号': 'lend_request_id', '放款时间': 'PassMth', '姓名': 'name', '手机号': 'mobile', '身份证号': 'id_no',
                 '银行卡号': 'card_no_pri'}
    base.rename(columns=base_dict, inplace=True)
    xx = base_addr(base).get_base_addrdata()
    yy = base_addr(base).get_base_var()

    with open(r'C:\Users\liyin\Desktop\FP_ABS\varall.pkl', 'rb') as f:
        varall = pickle.load(f)

    # ['bank_level_2.0'] 中国农业银行 训练模型时 样本中没有
    yy.to_csv(r'\\10.0.31.245\Ccx_Fp_ABS\lyk_A\data_0925\base_addr_var_2.csv', index=False)

    base_addr(base).bankcard_dict.lend_request_id.value_counts()
