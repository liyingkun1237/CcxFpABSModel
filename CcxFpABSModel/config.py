"""
配置文件所在位置

主要配置 日志存放位置，数据存储位置 等
"""
import os

# ProjectPATH = os.path.dirname(os.path.split(os.path.realpath(__file__))[0])
ProjectPATH = r'C:\Users\liyin\Desktop\CcxFpABSModel'
LOGFILEPATH = os.path.join(ProjectPATH, 'Log')
# 默认 为项目路径下
# 下一版优化到日志显示层级 联调时为详细版 上线后为简版

# 路径问题，上线之后，路径需要更改

FPABSDATABASEPATH = os.path.join(ProjectPATH, 'databases')  # 数据库存放文件目录
FPABSDATABASE = os.path.join(FPABSDATABASEPATH, 'FPABS.db')  # 存储数据的库名

# DATAFLIEPATH = os.path.join(ProjectPATH, 'Data')  # 原始输入数据,对应存库的表名
# VARFLIEPATH = os.path.join(ProjectPATH, 'Var')  # 计算的变量,对应存库的表名
# RESULTPATH = os.path.join(ProjectPATH, 'Result')  # 返回的结果,对应存库的表名
