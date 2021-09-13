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


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path', required=False, default='/usr/local/share/',
                                            help="This is the path where you would like to save the model. Default is the /usr/local/share"
                                            "automatically")
    parser.add_argument('-f', '--folds', nargs='+', default='None',
                                            help="folds to use for prediction. Default is Fold 4 adn 6"
                                            "automatically in the model output folder")
    parser.add_argument('-i', '--input_folder', help="Must contain all modalities for each patient in the correct"
                        " order (same as training). Files must be named "
                        "CASENAME_XXXX.nii.gz where XXXX is the modality "
                        "identifier (0000, 0001, etc)", required=True)
    parser.add_argument('-o', '--output_folder', required=True, help="folder for saving predictions")
    
    parser.add_argument('--all_in_gpu', type=str, default='None', required=False, help='can be None, False or True. '
                        "Do not touch.")
                        
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
    
    #parser.add_argument('-p', '--plans_identifier', help='do not touch this unless you know what you are doing',default=default_plans_identifier, required=False)

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


from nnunet.inference.predict import predict_from_folder
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
enablePrint()



def correct_header(input, output):
    img1 = nib.load(input)
    corr_affine = img1.get_qform()
    img1.set_sform(corr_affine)
    img1.update_header()
    img1.set_data_dtype(img1.get_data_dtype())
    nib.save(img1, output)

def correct_num_col(input, output):
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

def lpi_conform(input, output):
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

def download_model(parser):
    args = parser.parse_args()
    model_path = args.model_path
    folds = args.folds

    if folds == "None":
       url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_2f.zip'
    else:
       url = 'https://webdav-r3lab.uni.lu/public/deep_brain_seg/deep_brain_seg_model_7f.zip'

    r = requests.get(url, allow_redirects=True)
    model1 = model_path + 'deep_brain_seg_model.zip'
    open(model1 , 'wb').write(r.content)
    zip_ref = zipfile.ZipFile(model1 ,"r")
    zip_ref.extractall(model_path)
    os.remove(model1)

def inference(parser):
    blockPrint()
    args = parser.parse_args()
    directory = 'preprocessed'
    path = os.path.join(args.input_folder, directory)
    input_folder = path
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = 'None'
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    disable_tta = False
    step_size = 0.5
    overwrite_existing = False
    mode = "normal"
    all_in_gpu = args.all_in_gpu
    model = '3d_fullres'
    trainer_class_name = default_trainer
    cascade_trainer_class_name = default_cascade_trainer
    task_name = 'Task054_Mri'
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
    args = parser.parse_args()
    input = args.input_folder
    input_folder = input
    os.chdir(input)
    directory = 'preprocessed'
    path = os.path.join(input_folder, directory)
    if not os.path.exists(path):
        os.mkdir(path)
    for file in glob.glob('*.nii.gz'):
        input_img = os.path.join(input_folder, file)
        filename, file_extension = os.path.splitext(file)
        filename1, file_extension1 = os.path.splitext(filename)
        output_img = os.path.join(path, filename1 + '_0000'+ file_extension1 + file_extension)
        if not os.path.isfile(output_img):
           print('Pre-processing: ', file)
           correct_header(input_img, output_img)
           correct_num_col(output_img, output_img)
           lpi_conform(output_img, output_img)
        #else:
        #   print(file, ' is already preprocessed.')
    for file in glob.glob('*.nii'):
        input_img = os.path.join(input_folder, file)
        filename, file_extension = os.path.splitext(file)
        output_img = os.path.join(path, filename + '_0000'+ file_extension + '.gz')
        if not os.path.isfile(output_img):        
          print('Pre-processing: ', file)
          correct_header(input_img, output_img)
          correct_num_col(output_img, output_img)
          lpi_conform(output_img, output_img)
        #else:
        #   print(file, ' is already preprocessed.')

def main_preprocess():
    parser = arguments()
    preprocessing(parser)

def main_infer():
    parser = arguments()
    #set_paths(parser)
    args = parser.parse_args()
    folds = args.folds
    model_path = args.model_path
    model_path_second_part = 'model/nnUNet/3d_fullres/Task054_Mri/nnUNetTrainerV2__nnUNetPlansv2.1'
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

    print('Segmenting')
    blockPrint()
    inference(parser)
    enablePrint()

def main():
    main_preprocess()
    main_infer()


if __name__ == "__main__":
    main()
