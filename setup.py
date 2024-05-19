from setuptools import setup, find_packages

setup(
    name='vilg',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="ViLG",
    author='Kechun Xu',
    author_email='kcxu@zju.edu.cn',
    install_requires=[line for line in open('requirements.txt').readlines() if "@" not in line],
)