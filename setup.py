from distutils.core import setup

setup(
    name='CcxFpABSModel',
    version='0.1.1',  # 0.1.0
    packages=['CcxFpABSModel'],
    url='2018-03-15',  # 开启多进程
    license='ccx',
    author='liyingkun',
    author_email='liyingkun@ccx.cn',
    description='中诚信凡普ABS项目',
    package_data={'': ['*.py', 'exData/*.txt', 'exData/*.pkl']},
    data_files=[('', ['setup.py'])]
)
