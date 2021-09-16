# DBSegment <h1>
  
  This tool generates 30 deep brain structures segmentation, as well as a brain mask from T1-Weighted MRI. The whole procedure should take ~1 min for one case.
  
 The tool is available as a pip package. To run the package a GPU is required. 
  
 We highly recommend installing the package inside a virtual environment. For some instruction on virtual envrionments and pip package installation, please refer to: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

**Installation**
  
  `pip install DBSegment`
   
  Once the package is installed, you can get the segmention by running the following command:
 
  
**Example** 
  
  `DBSegment -i input_folder -o output_folder -mp path_to_model`
  
  Once it is done, two folders will be created, preprocessed_v3 and an output folder. The output folder contains the segmentation of the the 30 brain structures and the rest of the brain, filename.nii.gz, and a brain mask, filename_brainmask.nii.gz. The ouput files should be applied on the preprocessed image in the preprocessed folder, filename_0000.nii.gz.
  
 **Flags** 
  
  **-i**  is the input folder where your MR images are located. The input folder should contain nifti format T1 weighted MRI in *"*.nii.gz"* or *"*.nii"* format.

 `-i /Users/mehri.baniasadi/Documents/mr_data`

**-o**  is the output folder where the model outputs the segmentations.

 `-o /Users/mehri.baniasadi/Documents/mr_seg`

**-mp**  is the path to save the model. The default is /usr/local/share

  `-mp /Users/mehri.baniasadi/Documents/models`

**-f**  are the folds (networks) used for segmentation. The available folds are *0, 1, 2, 3, 4, 5, 6*. The default folds are *4* and *6*. We recommend to keep the default settings, and do not define this parameter.

  `-f 4 6`
  
  **-v**  is the the version of the preprocessing you would like to aply before segmenation. The default is v3 (LPI oritnation, 1mm voxel spacing, 256 Dimension). The alternative option is v1 (LPI orientaiton). Please note that by chaning the version to v1 the segmenation quality will reduce by 1-2%.

  `-v v1`
  
  **--disable_tta**
  This Flag is for the test time augmentation. The default is True and tta is disabled, to enable the tta, set this flag to True. By setting the flag to True, the segmenation quality will improve by ~0.2%, and the inference time will increase by 10-20 seconds.

  `--disable_tta True`
