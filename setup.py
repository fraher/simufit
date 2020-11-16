import os
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
   name='simufit',   
   version=os.getenv('Build.BuildNumber'),
   description='Simufit is a library designed to assist in the identification of distribution types for a given sampling of data.',
   author='Nate Bartlett and Christopher J Fraher',
   author_email='nbartlett3@gatech.edu;cfraher3@gatech.edu',   
   install_requires=required,   
   packages=["simufit"]
)