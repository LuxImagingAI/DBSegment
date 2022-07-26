from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.1.8'
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
                        "numpy==1.21.6",
                        "nibabel==3.2.2",
                        "scipy==1.7.3",
                        "batchgenerators==0.24",
                        "torch==1.11.0",
                        "tqdm==4.64.0",
                        "dicom2nifti==2.3.3",
                        "scikit-image==0.19.2",
                        "medpy==0.4.0",
                        "sklearn==0.0",
                        "SimpleITK==2.1.1.2",
                        "pandas==1.3.5",
                        "requests==2.27.1",
                        "tifffile==2021.11.2", 
                        "matplotlib==3.5.2",
                        "antspyx==0.3.2"],
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
