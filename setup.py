from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras','h5py','google-cloud-storage']

setup(
    name='experiment',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='ANRL Package',
)