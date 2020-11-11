from setuptools import setup

setup(
   name='simufit',
   version='0.0.1',
   description='Simufit is a library designed to assist in the identification of distribution types for a given sampling of data.',
   author='Nate Bartlett and Christopher J Fraher',
   author_email='nbartlett3@gatech.edu;cfraher3@gatech.edu',
   packages=['simufit'],  #same as name
   install_requires=[] # install_requires=['bar', 'greek'], #external packages as dependencies
)