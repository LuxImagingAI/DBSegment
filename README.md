# DBSegment <h1>

## Command line tool 
This tool generates 30 deep brain structures segmentation, as well as a brain mask from T1-Weighted MRI. The whole procedure should take ~1 min for one case.
For a defintion of the resulting labels refer to the paper or the provided ITK labels file `labels.txt`.
  
The tool is available as a pip package. **The package works on both GPU and CPU.**
  
We highly **recommend installing the package inside a virtual environment**. For some instruction on virtual envrionment and pip package installation, please refer to: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/.  
We also strongly recommend to work with two dedicated and separate _input_ and _output_ folders inside the project folder. This avoids any possible naming conflict.
  
We provide different ways of installation, at the moment: a pip package or docker image. Alternativly you could always choose to "manually" install by cloning the git repository and managing all dependencies by yourself. The key dependency is the [nnU-Net Framework](https://github.com/MIC-DKFZ/nnUNet) (and the super-depencies arising from that) as we use an nnU-Net segmentation model.
To use DBSegment in high performance computing environments (HPC), we recommend the pip package, as docker is not well suited for conventional multi-user envioronments.

  **Installation using Docker**
  
 1. Install Docker.
 2. Pull the DBSegment image `docker pull mehrib/dbsegment:v4`. You Need to do this just the first time. 
 3. Run the image `docker run -v "/input_folder/:/input/" -v "/output_folder/output/:/output/" mehrib/dbsegment:v4`
  
  
**Installation using pip**
  
  `pip install DBSegment`
   
  Once the package is installed, you can get the segmention by running the following command:
 
  
**Example** 
  
  `DBSegment -i input_folder -o output_folder -mp path_to_model`
  
  The input folder should contain you input image, e.g. *filename.nii.gz*. Once it is done, two folders will be created, a preprocessed and an output folder. The output folder contains the segmentations of the the 30 brain structures and one label for the rest of the brain, *filename.nii.gz*, a file containing 30 brain structures segmenation, *filename_seg.nii.gz*, and a brain mask, *filename_brainmask.nii.gz*. The ouput files should be applied on the preprocessed image in the preprocessed folder, *filename_0000.nii.gz*.
  
 **Parameters** 

  **-i** input folder where your MR images are located. The input folder should contain nifti format T1 weighted MRI in *"*.nii.gz"* or *"*.nii"* format.

 `-i /Users/mehri.baniasadi/Documents/mr_data`

**-o** output folder where the model outputs the segmentations.

 `-o /Users/mehri.baniasadi/Documents/mr_seg`

optional: **-mp** path to store the model that will be downloaded automatically. The default is /usr/local/share. 

  `-mp /Users/mehri.baniasadi/Documents/models`

optional: **-f** folds (networks) used for segmentation inferrence. The available folds are *0, 1, 2, 3, 4, 5, 6*. The default folds are *4* and *6*.   Using more folds will increase the needed computation time but potentially improve segmentation quality.
  
 optional: **-v**  is the the version of the preprocessing applied before segmenation. The default is v3 (LPI orienatation, 1mm voxel spacing, 256 Dimension). The alternative option avialable is v1 (only conforming images to LPI orientation, no further preprocessing). Please note that by changing the version to v1 the segmenation quality might be slightly altered. Note that in each case nnU-Net applies further preprocessing steps before the CNN model is invoked.

  `-v v3`
  
  optional: **--disable_tta**
  This Flag is for the test time augmentation. The default is True and tta is disabled, to enable the tta, set this flag to True. By setting the flag to True, the segmenation quality will improve by ~0.2%, and the inference time will increase by 10-20 seconds.

  `--disable_tta True`

## Model  
We provide the trained neural network model for download. The command line tool will automatically download the model files if not present at the first run.

## How to cite 
  You can find much more information, in particular on robustness across acuqisition domains in the following paper. Please cite this publication when using the tool in own works:

> Mehri Baniasadi, Mikkel V. Petersen, Jorge Goncalves, Vanja Vlasov, Andreas Horn, Frank Hertel, Andreas Husch (2021): Fast and robust segmentation of deep brain structures: Evaluation of transportabilityacross acquisition domain
