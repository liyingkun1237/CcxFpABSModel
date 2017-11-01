"""
此脚本为记录了对字段的脱敏逻辑

总结一下，需要脱敏的字段有

["loan_purpose","gender","education","marriage","house_property_type",
"resource_type","lend_company_type","job_title","department","car_property_type",
"iNatives","secret_to_family","Work_unit"
]
"""

import numpy as np


# loan_purpose
def f_dict_loan_purpose(x):
    '''
    # loan_purpose
    :return:
    '''
    purpose_dict = {'副业经营': 'sideline',
                    '装修': 'renovation',
                    '购车': 'buycar',
                    '购房': 'buyhouse',
                    '家电/数码产品消费': 'digital',
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


def f_dict_loan_purpose_1(x):
    '''
    # loan_purpose
    :return:
    '''
    purpose_dict = {'sideline': 1,
                    'renovation': 2,
                    'buycar': 3,
                    'buyhouse': 4,
                    'digital': 5,
                    'other': 6,
                    'capitalturnover': 7,
                    'education': 8}

    if x in purpose_dict.keys():
        return purpose_dict[x]
    elif x not in purpose_dict.keys():
        return purpose_dict['其他']
    else:
        return np.nan


# gender
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


# education
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


# marriage
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


# house_property_type
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


# resource_type
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


# lend_company_type
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


# job_title
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


# department
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


# car_property_type
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


# iNatives  secret_to_family
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


# Work_unit
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


def f_antiLogic(fpdata):
    '''
    脱敏的总调用函数
    :param fpdata:
    :return:
    '''
    fp_apply_data = fpdata.assign(
        # 类别转换 类别归约
        loan_purpose=lambda x: x.loan_purpose.apply(f_dict_loan_purpose),
        gender=lambda x: x.gender.apply(f_dict_gender),
        education=lambda x: x.education.apply(f_dict_education),
        marriage=lambda x: x.marriage.apply(f_dict_marriage),
        iNatives=lambda x: x.iNatives.apply(f_dict_bio),
        house_property_type=lambda x: x.house_property_type.apply(f_dict_house_property),
        resource_type=lambda x: x.resource_type.apply(f_dict_resource_type),
        lend_company_type=lambda x: x.lend_company_type.apply(f_dict_lend_company),
        job_title=lambda x: x.job_title.apply(f_dict_job_title),
        department=lambda x: x.department.apply(f_dict_department),
        car_property_type=lambda x: x.car_property_type.apply(f_dict_car_property),
        secret_to_family=lambda x: x.secret_to_family.apply(f_dict_bio),
        # work_unit_cate=lambda x: x.Work_unit.apply(self.f_work_uint),
        Work_unit=lambda x: x.Work_unit.apply(f_work_uint)
    ).assign(
        loan_purpose=lambda x: x.loan_purpose.apply(f_dict_loan_purpose_1),
    )

    return fp_apply_data
