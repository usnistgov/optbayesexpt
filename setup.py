from setuptools import setup, find_packages

# Fetches description from README.rst file
with open('M:\Software\OptBayesExptDesign\README.rst', "r") as f:
    long_description = f.read()

setup(name='optbayesexpt',
      version='0.1.7',
      description="Optimal Bayesian Experimental Design",
      long_description=long_description,
      classifiers=[
          "Intended Audience :: Science/Research",
          "License :: Public Domain",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3.6",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          'Development Status :: 4 - Beta'
      ],
      keywords='bayesian measurement physics experimental design',
      url='https://github.com/usnistgov/optbayesexpt/',
      author='Bob McMichael',
      author_email='rmcmichael@nist.gov',
      packages=find_packages()
      )
