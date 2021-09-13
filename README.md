# Deep Brain Segment <h1>
  
  This tool generates 30 deep brain structures segmentation, as well as a brain mask from T1-Weighted MRI. 
 The tool is available as a pip package.
  
  `pip install deep-brain-segment`
   
  The input folder should contain nifti format T1 weighted MRI in *"*.nii.gz"* or *"*.nii"* format.
  Once the package is installed, you can get the segmention by running the following command:
 
  
**Example** 
  
  `dbsegment -i input_folder -o output_folder -mp path_to_model`
  
 **Parameters** 
  
  **-i** input folder where your MR images are located. 

 `-i /Users/mehri.baniasadi/Documents/mr_data`

**-o** output folder where the model outputs the segmentations.

 `-o /Users/mehri.baniasadi/Documents/mr_seg`

optional: **-mp** path to the trained model. The default is /usr/local/share

  `-mp /Users/mehri.baniasadi/Documents/models`

optional: **-f** folds (networks) used for segmentation inferrence. The available folds are *0, 1, 2, 3, 4, 5, 6*. The default folds are *4* and *6*. We recommend to keep the default settings, and do not define this parameter. Using more folds will increase the needed computation time.

  `-f 4 6`
