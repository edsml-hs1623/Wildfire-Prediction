from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
setup(
    name="wildfire",  # Package name must be mcsim.
    # Extend setup.py file as required.
    version='1.0.0',  # Version number, required
    # packages=[''],# directories to install, required

    packages=find_packages(),
    include_package_data=True,

    install_requires = requirements,
    description='The goal of this project is to develop a comprehensive system that predicts a wildfire behavior using Recurrent Neural Networks (RNN), Generative AI, and data assimilation techniques. This system will use historical wildfire data, and satellite imagery feeds, to enhance prediction accuracy and provide actionable insights.',  
    url='https://github.com/ese-msc-2023/acds3-wildfire-rush',  
    author='Rush',  
)

