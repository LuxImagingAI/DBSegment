
DeepBrainSegement
Mehri Baniasadi - 2021

This function create 30 deep brain structures segmentation, as well as a brain mask from T1-Weighted MRI

The input folder should contain nifti format T1W MRI in "*.nii.gz" or "*.nii" format

Example:

pip install DeepBrainSeg

dbsegment -i input_folder -o output_folder -mp path_to_model


Inputs:

-i is the input folder where you MR images are located. 
E.g. -i /Users/mehri.baniasadi/Documents/mr_data

-o is the output folder where you want the results to be saved.
e.g. -o /Users/mehri.baniasadi/Documents/mr_seg

-mp is the path where you would like the model folder to be saved. The default is /us/local/share
e.g. -mp /Users/mehri.baniasadi/Documents/models

-f are the folds (networks) you would like to be used for segmentation. The available folds are 0, 1, 2, 3, 4, 5, 6. The default folds are 4 and 6. We recommend to keep the default settings, and do not define this parameter.
e.g. -f 4 6





