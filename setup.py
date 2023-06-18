from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.2.0'
DESCRIPTION = 'A deep learning based method to segment deep brain structures from T1w MRI'
# Setting up
setup(
      name="DBSegment",
      version=VERSION,
      author="Mehri",
      author_email="mehri.baniasadi92@gmail.com",
      description=DESCRIPTION,
      packages=find_packages(),
      install_requires=[
                        "numpy==1.24.2",
                        "nibabel==5.1.0",
                        "scipy==1.10.1",
                        "batchgenerators==0.25",
                        "torch==2.0.0",
                        "tqdm==4.65.0",
                        "dicom2nifti==2.4.8",
                        "scikit-image==0.20.0",
                        "MedPy==0.4.0",
                        "scikit-learn==1.2.2",
                        "SimpleITK==2.2.1",
                        "pandas==2.0.0",
                        "requests==2.28.2 ",
                        "tifffile==2023.3.21", 
                        "matplotlib==3.7.1",
                        "antspyx==0.3.8",
                        "nnunet==1.7.1",
                        ],
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
