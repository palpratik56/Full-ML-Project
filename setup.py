from setuptools import find_packages, setup
from typing import List

def get_req(file_path:str)->List[str]:
    pkgs = []
    hyp = '-e .'
    with open (file_path) as  obj:
        pkgs = obj.readlines()
        pkgs = [r.replace('\n','') for r in pkgs]

        if hyp in pkgs:
            pkgs.remove(hyp)
    return pkgs

setup( name='MLproject',
    version='1.0.1',
    description='An end-to-end machine learning project with industry level project structure',
    author='PratikPal',
    author_email='palpratik56@gmail.com',
    packages=find_packages(),
    install_requires= get_req('requirements.txt') )