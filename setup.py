from setuptools import setup, find_packages
import os


# Fetches description from README.rst file
with open(os.path.join(os.getcwd(), 'README.md'), "r") as f:
    long_description = f.read()

setup(name='optbayesexpt',
      version='1.2.0',
      description="Optimal Bayesian Experimental Design",
      long_description=long_description,
      classifiers=[
          "Intended Audience :: Science/Research",
          "License :: Public Domain",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development :: Libraries :: Python Modules",
          'Development Status :: 4 - Beta'
      ],
      install_requires=['numpy', 'scipy', 'matplotlib'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      keywords='bayesian measurement physics experimental design',
      url='https://github.com/usnistgov/optbayesexpt/',
      author='Bob McMichael',
      author_email='rmcmichael@nist.gov',
      packages=find_packages()
      )
