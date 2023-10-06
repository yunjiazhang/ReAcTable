import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()
    
setuptools.setup(
    name='tabqa',
    version='0.1.0',
    author='Yunjia Zhang',
    author_email='yunjia@cs.wisc.edu',
    description='GPT-TabQA project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    packages=['tabqa'],
    install_requires=required,
)