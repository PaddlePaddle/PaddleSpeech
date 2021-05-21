#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/15
"""
语音处理工具箱。
生成whl格式安装包：python setup.py bdist_wheel

直接上传pypi：python setup.py sdist upload

用twine上传pypi：
生成安装包：python setup.py sdist
上传安装包：twine upload dist/phkit-0.0.3.tar.gz

注意：需要在home目录下建立.pypirc配置文件，文件内容格式：
[distutils]
index-servers=pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username: admin
password: admin
"""

from setuptools import setup, find_packages
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(os.path.splitext(os.path.basename(__name__))[0])
install_requires = ['pypinyin>=0.41.0', 'hanziconv', 'jieba>=0.42.1', 'tqdm', 'inflect', 'unidecode']
requires = install_requires


def create_readme():
    from phkit import readme_docs
    docs = []
    with open("README.md", "wt", encoding="utf8") as fout:
        for doc in readme_docs:
            fout.write(doc)
            docs.append(doc)
    return "".join(docs)


def pip_install():
    for pkg in install_requires + requires:
        try:
            os.system("pip install {}".format(pkg))
        except Exception as e:
            logger.info("pip install {} failed".format(pkg))


pip_install()
phkit_doc = create_readme()

from phkit import __version__ as phkit_version

setup(
    name="phkit",
    version=phkit_version,
    author="kuangdd",
    author_email="kuangdd@foxmail.com",
    description="phoneme toolkit",
    long_description=phkit_doc,
    long_description_content_type="text/markdown",
    url="https://github.com/KuangDD/phkit",
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=install_requires,  # 指定项目最低限度需要运行的依赖项
    python_requires='>=3.5',  # python的依赖关系
    package_data={
        'txt': ['requirements.txt'],
        'md': ['**/*.md', '*.md'],
    },  # 包数据，通常是与软件包实现密切相关的数据
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
    ],
)

if __name__ == "__main__":
    print(__file__)
