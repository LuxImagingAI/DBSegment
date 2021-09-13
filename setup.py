from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.4'
DESCRIPTION = 'A deep learnign based method to segment deep brain structures from T1w MRI'
# Setting up
setup(
      name="deep-brain-segment",
      version=VERSION,
      author="Mehri",
      author_email="mehri.baniasadi92@gmail.com",
      description=DESCRIPTION,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      install_requires=[
                        "numpy==1.17.4",
                        "nibabel==2.5.1",
                        "scipy==1.3.1",
                        "nnunet-inference-on-cpu-and-gpu==1.6.6",
                        "batchgenerators==0.21"],
      entry_points={
      'console_scripts': [
                          "dbsegment = deep_brain_segment.deep_brain_segment:main",
                          "dbsegment_prep = deep_brain_segment.deep_brain_segment:main_preprocess",
                          "dbsegment_infer = deep_brain_segment.deep_brain_segment:main_infer",
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
