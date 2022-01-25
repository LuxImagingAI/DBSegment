# DBSegment <h1>

  ## command line tool 
  This tool generates 30 deep brain structures segmentation, as well as a brain mask from T1-Weighted MRI. The whole procedure should take ~1 min for one case.
  For a defintion of the resulting labels refer to the paper or the provided ITK labels file `labels.txt`.
  
 The tool is available as a pip package. **To run the package a GPU is required.**
  
 We highly recommend installing the package inside a virtual environment. For some instruction on virtual envrionment and pip package installation, please refer to: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

**Installation**
  
  `pip install DBSegment`
   
  Once the package is installed, you can get the segmention by running the following command:
 
  
**Example** 
  
  `DBSegment -i input_folder -o output_folder -mp path_to_model`
  
  The input folder should contain you input image, e.g. *filename.nii.gz*. Once it is done, two folders will be created, a preprocessed and an output folder. The output folder contains the segmentations of the the 30 brain structures and one label for the rest of the brain, *filename.nii.gz*, a file containing 30 brian structures segmenation, *filename_seg.nii.gz*, and a brain mask, *filename_brainmask.nii.gz*. The ouput files should be applied on the preprocessed image in the preprocessed folder, *filename_0000.nii.gz*.
  
 **Parameters** 

  **-i** input folder where your MR images are located. The input folder should contain nifti format T1 weighted MRI in *"*.nii.gz"* or *"*.nii"* format.

 `-i /Users/mehri.baniasadi/Documents/mr_data`

**-o** output folder where the model outputs the segmentations.

 `-o /Users/mehri.baniasadi/Documents/mr_seg`

optional: **-mp** path to the trained model. The default is /usr/local/share

  `-mp /Users/mehri.baniasadi/Documents/models`

optional: **-f** folds (networks) used for segmentation inferrence. The available folds are *0, 1, 2, 3, 4, 5, 6*. The default folds are *4* and *6*. We recommend to keep the default settings, and do not define this parameter. Using more folds will increase the needed computation time.
  
  **-v**  is the the version of the preprocessing you would like to aply before segmenation. The default is v3 (LPI oritnation, 1mm voxel spacing, 256 Dimension). The alternative option is v1 (LPI orientaiton). Please note that by chaning the version to v1 the segmenation quality will reduce by 1-2%.

  `-v v1`
  
  **--disable_tta**
  This Flag is for the test time augmentation. The default is True and tta is disabled, to enable the tta, set this flag to True. By setting the flag to True, the segmenation quality will improve by ~0.2%, and the inference time will increase by 10-20 seconds.

  `--disable_tta True`

## Model  
We provide the trained neural network model for download. The command line tool will automatically download the model files if not present at the first run.

## How to cite 
  You can find much more information, in particular on robustness across acuqisition domains in the following paper. Please cite this publication when using the tool in own works:

> Mehri Baniasadi, Mikkel V. Petersen, Jorge Goncalves, Vanja Vlasov, Andreas Horn, Frank Hertel, Andreas Husch (2021): Fast and robust segmentation of deep brain structures: Evaluation of transportabilityacross acquisition domain
