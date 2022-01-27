from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.1.3'
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
                        "batchgenerators==0.21",
                        "torch>=1.6.0a",
                        "tqdm",
                        "dicom2nifti",
                        "scikit-image>=0.14",
                        "medpy",
                        "sklearn",
                        "SimpleITK",
                        "pandas",
                        "requests",
                        "tifffile", 
                        "matplotlib",],
      entry_points={
      'console_scripts': [
                          "DBSegment = DBSegment.DBSegment:main",
                          "DBSegment_prep = DBSegment.DBSegment:main_preprocess",
                          "DBSegment_infer = DBSegment.DBSegment:main_infer",
                          "DBSegment_bm = DBSegment.DBSegment:main_brainmask",
                          'nnUNet_convert_decathlon_task = DBSegment.nnunet.experiment_planning.nnUNet_convert_decathlon_task:main',
                          'nnUNet_plan_and_preprocess = DBSegment.nnunet.experiment_planning.nnUNet_plan_and_preprocess:main',
                          'nnUNet_train = DBSegment.nnunet.run.run_training:main',
                          'nnUNet_train_DP = DBSegment.nnunet.run.run_training_DP:main',
                          'nnUNet_train_DDP = DBSegment.nnunet.run.run_training_DDP:main',
                          'nnUNet_predict = DBSegment.nnunet.inference.predict_simple:main',
                          'nnUNet_ensemble = DBSegment.nnunet.inference.ensemble_predictions:main',
                          'nnUNet_find_best_configuration = DBSegment.nnunet.evaluation.model_selection.figure_out_what_to_submit:main',
                          'nnUNet_print_available_pretrained_models = DBSegment.nnunet.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
                          'nnUNet_print_pretrained_model_info = DBSegment.nnunet.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
                          'nnUNet_download_pretrained_model = DBSegment.nnunet.inference.pretrained_models.download_pretrained_model:download_by_name',
                          'nnUNet_download_pretrained_model_by_url = DBSegment.nnunet.inference.pretrained_models.download_pretrained_model:download_by_url',
                          'nnUNet_determine_postprocessing = DBSegment.nnunet.postprocessing.consolidate_postprocessing_simple:main',
                          'nnUNet_export_model_to_zip = DBSegment.nnunet.inference.pretrained_models.collect_pretrained_models:export_entry_point',
                          'nnUNet_install_pretrained_model_from_zip = DBSegment.nnunet.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
                          'nnUNet_change_trainer_class = DBSegment.nnunet.inference.change_trainer:main',
                          'nnUNet_evaluate_folder = DBSegment.nnunet.evaluation.evaluator:nnunet_evaluate_folder',
                          'nnUNet_plot_task_pngs = DBSegment.nnunet.utilities.overlay_plots:entry_point_generate_overlay',
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
