"""
更新银行卡字典

#1.从三方数据获取到的银行卡归属地信息 直接转为pkl,供单次使用
#2.等到调用结束后，将数据保存时更新到总的pkl里

"""
import os
import pickle
from CcxFpABSModel.log import ABS_log

@ABS_log('FPABS')
def f_updateBankDict(bank):
    '''
    更新单条数据集为模型使用
    :param bank:
    :return:
    '''
    file_path = os.path.split(os.path.realpath(__file__))[0]
    bank_path = os.path.join(file_path, 'exData', 'bankAddr_dict_One.pkl')
    # bank_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bankAddr_dict_All.pkl'

    bank = bank.rename(columns={'province': 'bank_prov', 'city': 'bank_city'})

    with open(bank_path, 'wb') as f:
        pickle.dump(bank, f)

    # with open(bank_path, 'rb') as f:
    #     bankcard_dict = pickle.load(f)

    # print('bankData update')


if __name__ == '__main__':
    # import pickle

    bank_path = r'C:\Users\liyin\Desktop\FP_ABS\bank_addr_dict_all.pkl'
    with open(bank_path, 'rb') as f:
        bankcard_dict = pickle.load(f)

    bankcard_dict

    with open(r'C:\Users\liyin\Desktop\FP_ABS\bank_addr_dict_3rd.pkl', 'rb') as f:
        bankcard_dict_2 = pickle.load(f)

    ##############################重新制作bank的字典
    import pandas as pd

    bank_2 = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bank_2.txt', sep='\t')
    bank_2 = bank_2[['lend_request_id', 'PassMth', 'card_no_pri', 'bank', 'cardkind',
                     'cardtype', 'province', 'city']]

    # bank_2.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bank_2.csv', index=False)

    bank_1 = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bank_1.txt', sep='\t')
    bank_3 = pd.read_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bank_3.txt', sep='\t')

    pd.concat([bank_1, bank_2, bank_3]).drop_duplicates().lend_request_id.value_counts()
    # 326833,301399,327268

    bank_all = pd.concat([bank_1, bank_2, bank_3]).drop_duplicates('lend_request_id')

    bank_all = bank_all.rename(columns={'province': 'bank_prov', 'city': 'bank_city'})

    bank_all.to_csv(r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bank_all.csv', index=False)

    ############################
    bank_path = r'C:\Users\liyin\Desktop\CcxFpABSModel\bank\bankAddr_dict_All.pkl'

    with open(bank_path, 'wb') as f:
        pickle.dump(bank_all, f)

    with open(bank_path, 'rb') as f:
        bankcard_dict = pickle.load(f)
