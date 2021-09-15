from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.3'
DESCRIPTION = 'A deep learnign based method to segment deep brain structures from T1w MRI'
# Setting up
setup(
      name="DBSegment",
      version=VERSION,
      author="Mehri",
      author_email="mehri.baniasadi92@gmail.com",
      description=DESCRIPTION,
      packages=find_packages(),
      install_requires=[
                        "numpy>=1.17.4",
                        "nibabel>=2.5.1",
                        "scipy>=1.3.1",
                        "nnunet-inference-on-cpu-and-gpu==1.6.6",
                        "batchgenerators==0.21"],
      entry_points={
      'console_scripts': [
                          "DBSegment = DBSegment.DBSegment:main",
                          "DBSegment_prep = DBSegment.DBSegment:main_preprocess",
                          "DBSegment_infer = DBSegment.DBSegment:main_infer",
                          "DBSegment_bm = DBSegment.DBSegment:main_brainmask",
                          ] },
      keywords=['python'],
      classifiers=[
                   "Development Status :: 1 - Planning",
                   "Intended Audience :: Developers",
                   "Programming Language :: Python :: 3",
                   "Operating System :: Unix",
                   "Operating System :: MacOS :: MacOS X",
                   "Operating System :: Microsoft :: Windows",
                   ]
      )
