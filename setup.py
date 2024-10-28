from setuptools import find_packages, setup

def get_reqs(requirements_path):
    'Return a list of requirements from the requirements text file'
    
    with open(requirements_path, 'r') as file:
        reqs=file.readlines()
        reqs=[req.replace('\n','') for req in reqs if req!='-e .']
    return reqs 

setup(
    name='brain_tumor_mri_tfx',
    version='0.0.1',
    packages=find_packages(),
    install_requires=get_reqs('requirements.txt')
)