import sys
import numpy as np
import nibabel as nib
import os
from nibabel.freesurfer.mghformat import MGHHeader
from scipy.ndimage import affine_transform
from numpy.linalg import inv
import glob
import requests
import zipfile
import argparse
import torch
import SimpleITK as sitk
import warnings
import platform


def arguments():       
     parser = argparse.ArgumentParser(description='Model parameters')
    
     parser.add_argument('-i', '--input_folder', help="Path to the folder containing the input T1w MRIs", required=True)
    
     parser.add_argument('-o', '--output_folder', required=True, help="This is the path where you would like to save the model predictions")

     if (platform.system() != 'Darwin'):
          default_path = '/usr/local/share/'
     else:
          default_path = 'os.path.expanduser("~")' # to avoid sudo restrictions
     parser.add_argument('-mp', '--model_path', required=False, default=default_path,
                                            help="This is the path where you would like to save the model. Default is the /usr/local/share for Linux or your user path on MacOS"
                                            "automatically")
                                            
     parser.add_argument('-v', '--version_of_preprocessing', required=False , default='v3',
                                                                help="You can set this to v1, to change only the orientation of the input image during the preprocessing."
                                                               "It conforms the input image to LPI (itksnap) orientation."
                                                               "This might reduce the quality of the segmentation by 1-2%."
                                                                "The default is v3, it confomrs the input image to orientation LPI (itksnap), 1mm voxel spacing, and 256 dimension.")
                                            
     parser.add_argument('-f', '--folds', nargs='+', default='None',
                                            help="folds to use for prediction. Default is Fold 4 adn 6."
                                            "You can choose any fold between 0 to 6")
                                            
     parser.add_argument('-n', '--convert_to_native', required=False, default=True, action="store_true",
                    help="Set this flag if the target folder contains predictions that you would like to overwrite"
                        "Default is False") 
                        
     parser.add_argument('-ss', "--sub_stn", type=int, required=False, default=0, help='This is to define which network to use for segmenation.'
                                            "Network 0 contains the STN as a whole structure."
                                            "Network 1 contains STN substrucures.")                                                                   

     parser.add_argument('--disable_tta', type=str, required=False, default='None',
                            help="set this flag to False to disable test time data augmentation via mirroring. Speeds up inference "
                            "Default is True")

     parser.add_argument('--overwrite_existing', required=False, default=False, action="store_true",
                    help="Set this flag if the target folder contains predictions that you would like to overwrite"
                        "Default is False")
    
     parser.add_argument('--all_in_gpu', type=str, default='None', required=False, help='can be None, False or True. '
                        "Do not touch, unless you have applied changes in nnunet for running the network on CPU only."
                         "In this case set this flasg to False")
                        
     parser.add_argument("--num_threads_nifti_save", required=False, default=2, type=int, help=
                                            "Determines many background processes will be used for segmentation export. Reduce this if you "
                                            "run into out of memory (RAM) problems. Default: 2")

     parser.add_argument('-z', '--save_npz', required=False, action='store_true',
                    help="use this if you want to ensemble these predictions with those of other models. Softmax "
                    "probabilities will be saved as compressed numpy arrays in output_folder and can be "
                    "merged between output_folders with nnUNet_ensemble_predictions")
    
     parser.add_argument('-l', '--lowres_segmentations', required=False, default='None',
                        help="if model is the highres stage of the cascade then you can use this folder to provide "
                        "predictions from the low resolution 3D U-Net. If this is left at default, the "
                        "predictions will be generated automatically (provided that the 3D low resolution U-Net "
                        "network weights are present")
        
     parser.add_argument("--part_id", type=int, required=False, default=0, help="Used to parallelize the prediction of "
                                            "the folder over several GPUs. If you "
                                            "want to use n GPUs to predict this "
                                            "folder you need to run this command "
                                            "n times with --part_id=0, ... n-1 and "
                                            "--num_parts=n (each with a different "
                                            "GPU (for example via "
                                            "CUDA_VISIBLE_DEVICES=X)")
                        
     parser.add_argument("--num_parts", type=int, required=False, default=1,
                                            help="Used to parallelize the prediction of "
                                            "the folder over several GPUs. If you "
                                            "want to use n GPUs to predict this "
                                            "folder you need to run this command "
                                            "n times with --part_id=0, ... n-1 and "
                                            "--num_parts=n (each with a different "
                                            "GPU (via "
                                            "CUDA_VISIBLE_DEVICES=X)")

     parser.add_argument("--num_threads_preprocessing", required=False, default=6, type=int, help=
                                            "Determines many background processes will be used for data preprocessing. Reduce this if you "
                                            "run into out of memory (RAM) problems. Default: 6")

     return parser


def set_paths(parser):
   
    args = parser.parse_args()
    model_path = args.model_path
    folds = args.folds
    if folds == "None":       
        os.environ['nnUNet_raw_data_base']= os.path.join(model_path,'deep_brain_seg_model_2f')
        os.environ['nnUNet_preprocessed']= os.path.join(model_path,'deep_brain_seg_model_2f/preprocess_nnUNet')
        os.environ['RESULTS_FOLDER']= os.path.join(model_path,'deep_brain_seg_model_2f/model')
    else:
        os.environ['nnUNet_raw_data_base']= os.path.join(model_path,'deep_brain_seg_model_7f') 
        os.environ['nnUNet_preprocessed']= os.path.join(model_path,'deep_brain_seg_model_7f/preprocess_nnUNet')
        os.environ['RESULTS_FOLDER']= os.path.join(model_path,'deep_brain_seg_model_7f/model')        

parser = arguments()
set_paths(parser)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()

import DBSegment.nnunet
from DBSegment.nnunet.inference.predict import predict_from_folder
from DBSegment.nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from DBSegment.nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
enablePrint()



def correct_header(input, output):
     """
        This function corrects the sforms, qform of the input image.
     """
     img1 = nib.load(input)
     corr_affine = img1.get_qform()
     img1.set_sform(corr_affine)
     img1.update_header()
     #img1.set_data_dtype(img1.get_data_dtype())
     img1.set_data_dtype(np.uint8)
     
     nib.save(img1, output)

def correct_num_col(input, output):
    """
        This function corrects the shape of the image. 
        e.g when it has the shape (1, 256, 256, 256) which can not be input to the preprocessing function.
    """
    img1 = nib.load(input)
    aff = img1.get_qform()
    mr_mat = img1.get_fdata()
    mr_shape = mr_mat.shape
    if len(mr_shape) > 3:
        l0 = mr_mat.shape[0]
        l1 = mr_mat.shape[1]
        l2 = mr_mat.shape[2]
        l3 = mr_mat.shape[3]
        new_mr = mr_mat.reshape(l0,l1,l2)
        new_mr = nib.Nifti1Image(new_mr, affine = aff)
        new_mr.set_data_dtype(img1.get_data_dtype())
        nib.save(new_mr, output)

def conform_v3(input, output):
    """
        This function conforms the images to the same orientation, LPI,
        Same voxel spacing 1mm x 1mm x 1mm,
        and the same dimension 256 x 256 x 256.
    """
    img = nib.load(input)
    h1 = MGHHeader.from_header(img)
    
    h1.set_data_shape([256, 256, 256])
    h1.set_zooms([1, 1, 1])
    
    h1['Mdc'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
    ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    vox2vox=inv(h1.get_affine())@ ras2ras @ img.affine
    
    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=h1.get_data_shape(), order=1)
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)
    
    new_img.set_data_dtype(img.get_data_dtype())
    nib.save(new_img, output)


def conform_v1(input, output):
    """
        This function conforms the images to the same orientation, LPI.
    """    
    img = nib.load(input)
    h1 = MGHHeader.from_header(img)
    x1, y1, z1=img.shape[:3]
    h1.set_data_shape([x1, y1, z1])
    
    sx, sy, sz=img.header.get_zooms()
    h1.set_zooms([sx, sy, sz])
    h1['Mdc'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]
    
    ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    vox2vox=inv(h1.get_affine())@ ras2ras @ img.affine
    
    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=h1.get_data_shape(), order=1)
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)
    
    new_img.set_data_dtype(img.get_data_dtype())
    nib.save(new_img, output)


def brainmask_extraction(input, output1):
    """
        This function extracts the brainmask from the networks output. 
    """
    img1 = sitk.ReadImage(input)
    brainmask = sitk.BinaryThreshold( img1, 1, 31, 1, 0 )
    sitk.WriteImage(brainmask, output1)
  
    #img2 = nib.load(input)
    #data = img2.get_fdata()
    #data[data==1]=0
    #h1 = MGHHeader.from_header(img2)
    #h1.set_data_shape([256, 256, 256])
    #h1.set_zooms([1, 1, 1])
    #h1['Mdc'] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #h1['Pxyz_c'] = img2.affine.dot(np.hstack((np.array(img2.shape[:3]) / 2.0, [1])))[:3]
    #labels = nib.MGHImage(data, h1.get_affine(), h1)
    #labels.set_data_dtype(img2.get_data_dtype())
    #nib.save(labels, output2)


def download_model(parser):
    """
        This function downloads the model. If no folds are set, it downloads fold 4 and 6.
        If fold is specified as a parameter, it downloads all 6 folds of the network.
        The model will be saved in the specified path in the -mp parameter, otherwise,
        the default path is '/usr/local/share/'
    """
    args = parser.parse_args()
    model_path = args.model_path
    folds = args.folds
    network_number = args.sub_stn
    if network_number == 0:    
        if folds == "None":
            url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_2f.zip'
        else:
            url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_7f.zip'
    elif network_number == 1:
        if folds == "None":
            url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_1_2f.zip'
        else:
            url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_1_7f.zip'
    else:
        print('Please set the network_number equal to 0 or 1. Other model numbers are not valid.')
            

    #if folds == "None":
    #   url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_2f.zip'
    #else:
    #   url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_7f.zip'

    r = requests.get(url, allow_redirects=True)
    model1 = model_path + 'deep_brain_seg_model.zip'
    open(model1 , 'wb').write(r.content)
    zip_ref = zipfile.ZipFile(model1 ,"r")
    zip_ref.extractall(model_path)
    os.remove(model1)

def inference(parser):
    blockPrint()
    args = parser.parse_args()
    version_of_preprocessing = args.version_of_preprocessing
    if version_of_preprocessing=='v3':
       directory = 'preprocessed_v3'
    elif version_of_preprocessing=='v1':
       directory = 'preprocessed_v1'
    input_folder = args.input_folder
    input_folder = os.path.abspath(input_folder)
    input_folder = os.path.join(input_folder, directory)
    output_folder = args.output_folder
    output_folder = os.path.abspath(output_folder)
    output_folder = os.path.join(output_folder, directory)
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = 'None'
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = args.disable_tta
    step_size = 0.5
    overwrite_existing = args.overwrite_existing
    mode = "normal"
    all_in_gpu = args.all_in_gpu
    model = '3d_fullres'
    
    trainer_class_name = default_trainer
    cascade_trainer_class_name = default_cascade_trainer
    network_number = args.sub_stn
    if network_number == 0:    
        task_name = 'Task054_Mri'
    else:
        task_name = 'Task062_Mri'
        
    args.plans_identifier = default_plans_identifier
    args.disable_mixed_precision = False
    args.chk = 'model_final_checkpoint'

    if not task_name.startswith("Task"):
        task_id = int(task_name)
        task_name = convert_id_to_task_name(task_id)

    assert model in ["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"], "-m must be 2d, 3d_lowres, 3d_fullres or " \
        "3d_cascade_fullres"
    

    if lowres_segmentations == "None":
        lowres_segmentations = None
    
    if isinstance(folds, list):
        if folds[0] == 'all' and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = [int(4), int(6)]
    else:
        raise ValueError("Unexpected value for argument folds")
    
    assert all_in_gpu in ['None', 'False', 'True']
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    assert disable_tta in ['None', 'False', 'True']
    if disable_tta == "None":
         disable_tta = True
    elif disable_tta == "True":
         disable_tta = True
    elif disable_tta == "False":
         disable_tta = False


    trainer = trainer_class_name
        
    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" +
                                 args.plans_identifier)
    assert isdir(model_folder_name), "model output folder not found. Expected: %s" % model_folder_name

    predict_from_folder(model_folder_name, input_folder, output_folder, folds, save_npz, num_threads_preprocessing,
                                                         num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                                                         overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                                                         mixed_precision=not args.disable_mixed_precision,
                                                         step_size=step_size, checkpoint_name=args.chk)
    enablePrint()

def preprocessing(parser):
    """
        add comments
    """
    args = parser.parse_args()
    input_folder = args.input_folder
    #input_folder = input
    input_folder = os.path.abspath(input_folder)
    overwrite_existing = args.overwrite_existing
    version_of_preprocessing = args.version_of_preprocessing
    #os.chdir(input_folder)
    if version_of_preprocessing=='v3':
        directory = 'preprocessed_v3'
    elif version_of_preprocessing=='v1':
        directory = 'preprocessed_v1'
    path = os.path.join(input_folder, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    for file in glob.glob(os.path.join(input_folder,'*.nii.gz')):
        file_path , file_name = os.path.split(file)
        input_img = os.path.join(input_folder, file_name)
        filename, file_extension = os.path.splitext(file_name)
        filename1, file_extension1 = os.path.splitext(filename)
        output_img = os.path.join(path, filename1 + '_0000'+ file_extension1 + file_extension)
        if not overwrite_existing:
           if not os.path.isfile(output_img):
               print('Pre-processing: ', file_name)
               correct_header(input_img, output_img)
               correct_num_col(output_img, output_img)
               if version_of_preprocessing=='v3':
                  conform_v3(output_img, output_img)
               elif version_of_preprocessing=='v1':
                  conform_v1(output_img, output_img)
        elif overwrite_existing:
               print('Pre-processing: ', file_name)
               correct_header(input_img, output_img)
               correct_num_col(output_img, output_img)
               if version_of_preprocessing=='v3':
                  conform_v3(output_img, output_img)
               elif version_of_preprocessing=='v1':
                  conform_v1(output_img, output_img)

        #else:
        #   print(file, ' is already preprocessed.')
    for file in glob.glob(os.path.join(input_folder,'*.nii')):
        file_path , file_name = os.path.split(file)
        input_img = os.path.join(input_folder, file_name)
        filename, file_extension = os.path.splitext(file_name)
        output_img = os.path.join(path, filename + '_0000'+ file_extension + '.gz')
        if not overwrite_existing:        
             if not os.path.isfile(output_img):        
                print('Pre-processing: ', file_name)
                correct_header(input_img, output_img)
                correct_num_col(output_img, output_img)
                if version_of_preprocessing=='v3':
                   conform_v3(output_img, output_img)
                elif version_of_preprocessing=='v1':
                   conform_v1(output_img, output_img)
        elif overwrite_existing:
             print('Pre-processing: ', file_name)
             correct_header(input_img, output_img)
             correct_num_col(output_img, output_img)
             if version_of_preprocessing=='v3':
                conform_v3(output_img, output_img)
             elif version_of_preprocessing=='v1':
                conform_v1(output_img, output_img)
        #else:
        #   print(file, ' is already preprocessed.')
        
def transfer_to_native(mr_prep_str, mr_native_str, seg_prep_str, seg_native_str):
    from ants import apply_transforms, registration, image_read
    import numpy
    seg_prep = image_read(seg_prep_str)
    mr_prep = image_read(mr_prep_str)
    mr_native = image_read(mr_native_str)
    #print(mr_native)
    #print('count nan values: ', np.sum(np.isnan(mr_native.numpy())))
    #print(mr_native.numpy())
    #new_array=mr_native.numpy()
    #new_array[np.isnan(new_array)]=0
    #print(new_array) 
    #mr_native.new_image_like(new_array)
    #print(mr_native.numpy())
    if (np.sum(np.isnan(mr_native.numpy())) > 0):
        print('Error: There are nan values in the original image. conversion to native is not possible.')
    else:        
        aff = registration(mr_native, mr_prep, "Affine" )
        mywarpedimage = apply_transforms( fixed = mr_native, 
                                       moving = seg_prep, 
                                       transformlist = aff['fwdtransforms'],
                                       interpolator  = 'genericLabel') 
        #mywarpedimage_bm = apply_transforms( fixed = mr_native, 
        #                               moving = brainmask_prep, 
        #                               transformlist = aff['fwdtransforms'],
        #                               interpolator  = 'genericLabel')                                       
        mywarpedimage.to_file(seg_native_str)
        #mywarpedimage_bm.to_file(brainmask_native_str)

            

def main_preprocess():
    """
        add comments
    """
    parser = arguments()
    preprocessing(parser)
    
def main_infer():
    """
        add comments
    """
    parser = arguments()
    args = parser.parse_args()
    folds = args.folds
    model_path = args.model_path
    network_number = args.sub_stn
    if network_number == 0:
        model_path_second_part = 'model/nnUNet/3d_fullres/Task054_Mri/nnUNetTrainerV2__nnUNetPlansv2.1'
    else:
        model_path_second_part = 'model/nnUNet/3d_fullres/Task062_Mri/nnUNetTrainerV2__nnUNetPlansv2.1'
    if folds == "None":
      model_file = os.path.join(model_path, 'deep_brain_seg_model_2f',model_path_second_part)
      if not os.path.exists(model_file):  
         print('Downloading the model. The model is 900MB, it might take a while.')
         download_model(parser)
      elif os.path.exists(model_file):
        print('Model exists.')
    else:
      model_file = os.path.join(model_path, 'deep_brain_seg_model_7f', model_path_second_part)
      if not os.path.exists(model_file):
         print('Downloading the model. The model is 3GB, it might take a while.')
         download_model(parser)
      elif os.path.exists(model_file):
        print('Model exists.')

    print('Segmenting.')
    blockPrint()
    inference(parser)
    

def main_brainmask():
    """
        add comments
    """
    print('Brain mask extraction.')
    parser = arguments()
    args = parser.parse_args()
    output_folder = args.output_folder
            
    output_folder = os.path.abspath(output_folder)
    overwrite_existing = args.overwrite_existing
    for file in glob.glob(os.path.join(output_folder,'*.nii.gz')):
        file_path , file_name = os.path.split(file)
        input = os.path.join(output_folder, file_name)
        filename, file_extension = os.path.splitext(file_name)
        filename1, file_extension1 = os.path.splitext(filename)
        output1 = os.path.join(output_folder, filename1 + '_brainmask'+ file_extension1 + file_extension)
        if not overwrite_existing:
             if  (not os.path.isfile(output1)) and ('_brainmask' not in filename1):
                brainmask_extraction(input, output1)
        elif overwrite_existing:
             brainmask_extraction(input, output1)
     
           
def main_to_native():
    
    parser = arguments()
    args = parser.parse_args()
    convert_to_native = args.convert_to_native
    if convert_to_native:
        print('Transforming to the native space.')
        input_folder = args.input_folder
        output_folder = args.output_folder
        overwrite_existing = args.overwrite_existing
        version_of_preprocessing = args.version_of_preprocessing
    
        if version_of_preprocessing=='v3':
            directory = 'preprocessed_v3'
        elif version_of_preprocessing=='v1':
            directory = 'preprocessed_v1'
    
        input_mr_native = os.path.abspath(input_folder)    
        input_mr_preprocessed = os.path.join(input_mr_native, directory)
        output_seg_preprocessed = os.path.abspath(output_folder)
        output_seg_preprocessed = os.path.join(output_seg_preprocessed, directory)
        output_seg_native = os.path.abspath(output_folder)
        if not os.path.exists(output_seg_native):
            os.mkdir(output_seg_native)

        for file in glob.glob(os.path.join(input_mr_native,'*.nii')):
            mr_native_str = file
            print(file)
            file_path , file_name = os.path.split(file)
            filename, file_extension = os.path.splitext(file_name)
            mr_prep_str = os.path.join(input_mr_preprocessed, filename + '_0000' + file_extension + '.gz')
            seg_prep_str = os.path.join(output_seg_preprocessed, filename + file_extension + '.gz')
            seg_native_str = os.path.join(output_seg_native, filename + file_extension + '.gz')
            #brainmask_prep_str = os.path.join(output_seg_preprocessed, filename + '_brainmask' + file_extension + '.gz')
            #brainmask_native_str = os.path.join(output_seg_native, filename + '_brainmask' + file_extension + '.gz')
                
            if not overwrite_existing:
                if  (not os.path.isfile(seg_native_str)):
                     transfer_to_native(mr_prep_str, mr_native_str, seg_prep_str, seg_native_str)
                elif overwrite_existing:
                     transfer_to_native(mr_prep_str, mr_native_str, seg_prep_str, seg_native_str)
                     
        
        for file in glob.glob(os.path.join(input_mr_native,'*.nii.gz')):
            mr_native_str = file
            file_path , file_name = os.path.split(file)
            filename, file_extension = os.path.splitext(file_name)
            filename1, file_extension1 = os.path.splitext(filename)
            mr_prep_str = os.path.join(input_mr_preprocessed,  filename1 + '_0000' + file_extension1 + file_extension)
            seg_prep_str = os.path.join(output_seg_preprocessed, filename1 + file_extension1 + file_extension)
            seg_native_str = os.path.join(output_seg_native, filename1 + file_extension1 + file_extension)
            #brainmask_prep_str = os.path.join(output_seg_preprocessed, filename1 + '_brainmask' + file_extension1 + file_extension)
            #brainmask_native_str = os.path.join(output_seg_native, filename1 + '_brainmask' + file_extension1 + file_extension)
                
            if not overwrite_existing:
                 if  (not os.path.isfile(seg_native_str)):
                        transfer_to_native(mr_prep_str, mr_native_str, seg_prep_str, seg_native_str)
            elif overwrite_existing:
                transfer_to_native(mr_prep_str, mr_native_str, seg_prep_str, seg_native_str)
    else:
       print('No conversion to native was requested. No native folder will be created.')
             


def main():
    """
        add comments
    """
    warnings.filterwarnings("ignore", message=r'(.*)(CUDA is not available)(.*)' )
    main_preprocess()
    main_infer()
    enablePrint()
    main_to_native()
    main_brainmask()

if __name__ == "__main__":
    main()

