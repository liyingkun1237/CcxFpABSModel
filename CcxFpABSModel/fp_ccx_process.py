# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:32:52 2017

@author: bjwangchao1
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from functools import reduce
import os


class Fp_ccx_dataprocess(object):
    def __init__(self):
        # fp_allver_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\fp_ccx_onlineTest1030\fp_ccx.pkl'
        file_path = os.path.split(os.path.realpath(__file__))[0]
        fp_allver_path = os.path.join(file_path, 'exData', 'fp_ccx.pkl')
        with open(fp_allver_path, 'rb') as f:
            self.fp_varall = pickle.load(f)

    # 高管信息
    @staticmethod
    def senior_data(data):
        fr_data = data[data['是否法人'] == 1]  # 法人
        nfr_data = data[data['是否法人'] == 2]  # 非法人
        fr_info = fr_data.groupby('lend_request_id')['注册资本'].max().reset_index().rename(columns={'注册资本': 'FR_REG'})
        nfr_info = nfr_data.groupby('lend_request_id')['注册资本'].max().reset_index().rename(columns={'注册资本': 'NFR_REG'})
        df = pd.merge(fr_info, nfr_info, on='lend_request_id', how='outer')
        return df

    # 股东信息
    @staticmethod
    def shareholder_data(data):
        def merge_reduce(x, y):
            return pd.merge(x, y, on='lend_request_id', how='left')

        #        try:
        #            data.loc[data['投资人类型']=='!','投资人类型']=-1
        #        except Exception:
        #            print('投资人类型字段无感叹号')
        #        data['投资人类型']=data['投资人类型'].astype(np.float64)
        credit_code = data.groupby('lend_request_id')[['统一信用代码']].agg('count').reset_index().rename(
            columns={'统一信用代码': 'credit_code_count'})
        #        invest_type=data.groupby('LEND_REQUEST_ID')[['投资人类型']].agg(np.max).reset_index().rename(columns={'投资人类型':'invest_type'})
        sh_info1 = data.groupby('lend_request_id')['企业名称股东'].count().reset_index().rename(
            columns={'企业名称股东': 'SH_ENT_count'})
        sh_info2 = data.groupby('lend_request_id')['认缴出资额'].sum().reset_index().rename(
            columns={'认缴出资额': 'AMT_SUBCONAM'})
        sh_info3 = data.groupby('lend_request_id')['实缴出资额'].sum().reset_index().rename(
            columns={'实缴出资额': 'AMT_ACTCONAM'})
        var_list = [credit_code, sh_info1, sh_info2, sh_info3]
        var = reduce(merge_reduce, var_list)
        var['sub_act_diff'] = var['AMT_ACTCONAM'] - var['AMT_SUBCONAM']
        return var

    # 是否董监高
    @staticmethod
    def is_directors_of_high(data1, data2):

        def is_or_no(x):
            s = list(x)
            if pd.notnull(s).sum() >= 1:
                return 1
            else:
                return 0

        df1 = data1.groupby('lend_request_id')['企业名称'].agg(is_or_no).reset_index().rename(
            columns={'企业名称': 'is_directors_of_high'})
        df2 = data2.groupby('lend_request_id')['企业名称股东'].agg(is_or_no).reset_index().rename(
            columns={'企业名称股东': 'is_directors_of_high'})
        df = pd.concat([df1, df2], axis=0, ignore_index=True).drop_duplicates('lend_request_id')
        return df

    '''
    消费能力数据可不用
    '''

    # 消费能力
    #    @staticmethod
    #    def consuming_ability(data):
    #        data.columns=['id_no','name','card_no_pri','mobile','LEND_REQUEST_ID','details','passmth','consuming_ability_score']
    #        return data[['LEND_REQUEST_ID','consuming_ability_score']]

    @staticmethod
    def f_transdate(x):
        '''
        日期格式转换 lyk 20171031新增
        :param x: 日期数据 字符串
        :return: 日期格式的 日期
        '''
        try:
            return datetime.strptime(x, '%Y年%m月%d日')
        except:
            return pd.to_datetime('')

    # 失信信息
    @staticmethod
    def lose_promise_data(data):
        data['立案时间失信'] = data['立案时间失信'].replace(
            ['40907', '41623', '41391', '41275', '41302', '41453', '41456', '41040', '41142'], '2015年11月11日')
        data['立案时间失信'] = data['立案时间失信'].apply(Fp_ccx_dataprocess().f_transdate)
        data['PassMth'] = pd.to_datetime(data['PassMth'])
        data = data.loc[data['立案时间失信'] < data['PassMth']]
        df = data.groupby('lend_request_id')['被执行人履行情况'].agg(['count', np.max]).reset_index().rename(
            columns={'count': 'shixing_times', 'amax': 'shixing_status'})
        return df

    # 执行信息（模糊匹配）
    @staticmethod
    def execute_vague_data(data):
        data.PassMth = pd.to_datetime(data.PassMth)
        data['立案时间'] = pd.to_datetime(data['立案时间'])
        data = data.loc[data['立案时间'] < data['PassMth']]
        data.drop_duplicates(inplace=True)
        df = data.groupby('lend_request_id')['涉案金额'].agg(['count', np.mean, np.max]).reset_index().rename(
            columns={'count': 'zhixing_vague_times', 'mean': 'zhixing_vague_meanAmount',
                     'amax': 'zhixing_vague_maxAmount'})
        return df

    # 执行信息（精确匹配）
    @staticmethod
    def execute_exact_data(data):
        data.drop_duplicates(inplace=True)
        data.PassMth = pd.to_datetime(data.PassMth)
        data['立案时间精确'] = data['立案时间精确'].apply(Fp_ccx_dataprocess().f_transdate)
        data = data.loc[data['立案时间精确'] < data['PassMth']]
        df = data.groupby('lend_request_id')['涉案金额精确'].agg(['count', np.mean, np.max]).reset_index().rename(
            columns={'count': 'zhixing_exact_times', 'mean': 'zhixing_exact_meanAmount',
                     'amax': 'zhixing_exact_maxAmount'})
        return df

    # 逾期催欠
    @staticmethod
    def overdue_cuiqian_data(data):
        df = data.groupby('lend_request_id')['欠款金额'].agg(['count', np.mean]).reset_index().rename(
            columns={'count': 'overdue_cuiqian_times', 'mean': 'overdue_cuiqian_meanAmount'})
        return df

    # 银行卡逾期
    @staticmethod
    def bank_overdue_data(data):
        df = data.groupby('lend_request_id')['是否银行卡逾期'].agg('count').reset_index().rename(
            columns={'是否银行卡逾期': 'bank_overdue_times'})
        return df

    # 小贷逾期
    @staticmethod
    def xiaodai_overdue_data(data):
        df = data.groupby('lend_request_id')['逾期天数'].agg(['count', np.max]).rename(
            columns={'count': 'xiaodaiOverdue_times', 'amax': 'xiaodaiOver_daymax'}).reset_index()
        return df

    # 可能信息泄露
    @staticmethod
    def info_leakage_data(data):

        def is_or_no(x):
            a = list(x)
            if pd.notnull(a).sum() >= 1:
                return 1
            else:
                return 0

        data.drop_duplicates(inplace=True)
        df = data.groupby('lend_request_id')['可能信息泄露'].agg(is_or_no).reset_index().rename(
            columns={'可能信息泄露': 'is_info_leakage'})
        return df

    # 学历信息
    @staticmethod
    def education_data(data):
        from datetime import datetime
        now = datetime.now().year
        data['interval_date_guaduation'] = now - data['毕业时间']
        data.rename(columns={'学历': 'educational', '学历类型': 'education_type',
                             '毕业结论': 'guaduation_status'}, inplace=True)
        return data[
            ['lend_request_id', 'educational', 'education_type', 'guaduation_status', 'interval_date_guaduation']]

    # 核验信息
    def verf_data_pre(self, data1, data2):
        data = pd.concat([data1, data2], ignore_index=True)
        data['date'] = data.date.map(lambda x: str(x).split(' ')[0])
        data.drop_duplicates(inplace=True)
        verf_dict = {2010: '身份验证成功', 0: '处理成功', 9999: '系统错误', 2030: '验证成功', 2031: '验证失败', \
                     2063: '无姓名或身份证信息', 2061: '匹配失败', 2060: '匹配成功', 2062: '号码不存在', 2037: '该数据验证失败次数超过限制', \
                     1002: '账户余额不足', 2039: '暂不支持该数据验证', 2011: '身份验证失败 ', 2038: '数据格式错误', 2013: '输入信息不符合要求', \
                     2012: '库中无记录', 2001: '没有查询到结果', 1013: '验签失败', 1012: '参数为空或格式错误', 1003: '账户没有此接口访问权限'}
        data['result'] = data.RESULT.map(verf_dict)
        data['joggle'] = data.type.map(self.query_joggle)
        return data

    def count_notnull(self, x):
        a = set(x)
        count = 0
        for i in a:
            if pd.notnull(i):
                count += 1
        return count

    # 会包含np.nan
    def uni_count(self, x):
        return len(np.unique(x))

    def sucess_count(self, x):
        s = list(x)
        if (2030 in s) | (2060 in s) | (0 in s) | (2010 in s):
            return 1
        else:
            return 0

    def verffailed_count(self, x):
        s = list(x)
        if ('匹配失败' in s) | ('验证失败' in s) | ('身份验证失败' in s):
            return 1
        else:
            return 0

    def query_joggle(self, x):
        if x in [50, 19]:
            return '身份核验'
        if x in [67, 200, 75, 289, 223]:
            return '运营商核验'
        if x in [10, 110, 108]:
            return '卡三四要素核验'

    def verf_data(self, data):
        def merge_reduce(x, y):
            return pd.merge(x, y, left_index=True, right_index=True, how='left')

        df = data.groupby(['lend_request_id'])['type', 'INDUSTRYNAME', 'RESULT', 'result'].agg(
            {'type': 'count', 'INDUSTRYNAME': self.count_notnull, 'RESULT': self.sucess_count,
             'result': self.verffailed_count})
        df.rename(columns={'type': 'verf_times', 'INDUSTRYNAME': 'verf_institution_sum',
                           'RESULT': 'verf_is_success', 'result': 'verf_is_failed'}, inplace=True)
        card_verf = pd.DataFrame(
            data.query("joggle == '卡三四要素核验'").groupby('lend_request_id')['RESULT'].agg(self.sucess_count))
        card_verf.rename(columns={'RESULT': 'card_is_success'}, inplace=True)
        carr_operator = pd.DataFrame(
            data.query("joggle == '运营商核验'").groupby('lend_request_id')['RESULT'].agg(self.sucess_count))
        carr_operator.rename(columns={'RESULT': 'operation_is_success'}, inplace=True)
        uid_verf = pd.DataFrame(
            data.query("joggle == '身份核验'").groupby('lend_request_id')['RESULT'].agg(self.sucess_count))
        uid_verf.rename(columns={'RESULT': 'cid_is_success'}, inplace=True)
        bank = pd.DataFrame(
            data.query("INDUSTRYNAME == '银行'").groupby('lend_request_id')['type', 'RESULT'].agg(
                {'type': 'count', 'RESULT': self.sucess_count}))

        bank.rename(columns={'type': 'bank_verf_times', 'RESULT': 'bank_verf_sucess'}, inplace=True)
        consumer_finance = pd.DataFrame(
            data.query("INDUSTRYNAME == '消费金融机构'").groupby('lend_request_id')['type', 'RESULT'].agg(
                {'type': 'count', 'RESULT': self.sucess_count}))
        consumer_finance.rename(
            columns={'type': 'consumerfin_verf_times', 'RESULT': 'consumerfin_verf_sucess'}, inplace=True)
        other_institution = pd.DataFrame(
            data.query("INDUSTRYNAME != '消费金融机构' & INDUSTRYNAME != '银行'").groupby('lend_request_id')[
                'type', 'RESULT'].agg(
                {'type': 'count', 'RESULT': self.sucess_count}))  # lyk wc 1102更新 原因，数错了
        other_institution.rename(
            columns={'type': 'oth_institution_verf_times', 'RESULT': 'oth_institution_verf_sucess'},
            inplace=True)
        var_list = [df, card_verf, carr_operator, uid_verf, bank, consumer_finance, other_institution]
        var = reduce(merge_reduce, var_list)
        return var.reset_index()

    # 核验最终输出函数
    def verf_output_data(self, data1, data2):
        df = self.verf_data_pre(data1, data2)
        dfv = self.verf_data(df)
        return dfv

    # 评分信息
    def score_data(self, data1, data2):

        def is_score(x):
            s = list(x)
            if 0 in s:
                return 1
            else:
                return 0

        data = pd.concat([data1, data2], ignore_index=True)
        data['date'] = data.date_score.map(lambda x: str(x).split(' ')[0])
        data.drop_duplicates(inplace=True)
        score_dict = {2002: '信息不足，不予评分', 0: '处理成功', 1013: '验签失败', 1002: '账户余额不足'}
        data['result'] = data.RESULT_score.map(score_dict)

        df = data.groupby('lend_request_id')['type_score', 'RESULT_score', 'INDUSTRYNAME_score'].agg(
            {'type_score': 'count', 'RESULT_score': is_score, 'INDUSTRYNAME_score': self.count_notnull})
        df.rename(columns={'type_score': 'score_query_times', 'RESULT_score': 'score_pass',
                           'INDUSTRYNAME_score': 'score_institution_sum'}, inplace=True)
        df1 = df.reset_index()
        return df1

    @staticmethod
    def merge_data(data_list):

        def merge_reduce(x, y):
            return pd.merge(x, y, on='lend_request_id', how='left')

        from functools import reduce

        dd_var = reduce(merge_reduce, data_list)

        return dd_var

    def educational(self, x):
        edu_dict = {'专科': 1, '本科': 2, '硕士': 3, '博士': 4}
        if x in edu_dict.keys():
            return edu_dict[x]
        else:
            return np.nan

    def education_type(self, x):
        edu_type = {'成人': 1, '网络教育': 2, '自考': 3, '开放教育': 4, \
                    '普通': 5, '研究生': 6}
        if x in edu_type.keys():
            return edu_type[x]
        else:
            return np.nan

    def guaduation_status(self, x):
        status_dict = {'毕业': 1, '结业': 2}
        if x in status_dict.keys():
            return status_dict[x]
        else:
            return np.nan

    def shixing_status_f(self, x):
        shixing_dict = {'全部未履行': 1, '部分未履行': 2}
        if x in shixing_dict.keys():
            return shixing_dict[x]
        else:
            return np.nan

    def education_reduction(self, data):
        red_data = data.assign(
            educational=lambda x: x['educational'].apply(lambda s: self.educational(s)),
            education_type=lambda x: x['education_type'].apply(lambda s: self.education_type(s)),
            guaduation_status=lambda x: x['guaduation_status'].apply(lambda s: self.guaduation_status(s)),
            shixing_status=lambda x: x['shixing_status'].apply(lambda s: self.shixing_status_f(s)))

        return red_data

    def dummy_ccx_data(self, data):
        col = ['education_type', 'educational', 'guaduation_status', 'shixing_status']
        dummy_data = pd.get_dummies(self.education_reduction(data), columns=col, dummy_na=True)
        dummy_data['risk_times'] = dummy_data[
            ['shixing_times', 'zhixing_vague_times', 'zhixing_exact_times', 'overdue_cuiqian_times',
             'bank_overdue_times', 'xiaodaiOverdue_times']].sum(axis=1)
        return self.f_keep_fpdummy(col, dummy_data)[self.fp_varall.columns]

    def f_keep_fpdummy(self, col, var):
        '''

        :param col: 需要dummy的列名
        :param var: dummy后的数据集
        :return: 和原训练样本保持一致数据结构的数据集
        '''
        try:
            var.rename(columns={'LEND_REQUEST_ID': 'lend_request_id'}, inplace=True)
        except Exception as e:
            print(e)
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


if __name__ == "__main__":
    ccx_data = Fp_ccx_dataprocess()  # 生成对象
    # var_list=[ccx_data.education_data(education.head(1)),ccx_data.senior_data(gaoguan.head(1)),ccx_data.shareholder_data(gudong.head(1)),ccx_data.lose_promise_data(shixing.head(1)),\
    #       ccx_data.execute_vague_data(zhixing_vague.head(1)),ccx_data.execute_exact_data(zhixing_exact.head(1)),ccx_data.overdue_cuiqian_data(overdue_cuiqian.head(1)),\
    #      ccx_data.bank_overdue_data(bank_ovredue.head(1)),ccx_data.xiaodai_overdue_data(xiaodai_overdue.head(1)),ccx_data.info_leakage_data(info_leakage.head(1)),\
    #     ccx_data.is_directors_of_high(gaoguan.head(1),gudong.head(1)),ccx_data.verf_output_data(cid_verf,mob_verf.head(1)), ccx_data.score_data(cid_score.head(1),mob_score.head(1)) ]
    # fp_ccx_data=ccx_data.merge_data(var_list) #pre_onehot的数据
    # fp_ccx_data_dummy=ccx_data.dummy_ccx_data(fp_ccx_data)  #用于模型的数据
