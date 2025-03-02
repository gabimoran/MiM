from os import path
import numpy as np
from skimage import filters, measure
from scipy.ndimage import binary_fill_holes
# from PIL import Image, ImageFile
import cv2
import multiprocessing as mp 


		
def folder_path(dic, rute = ''): 
    ''' From a dict with the folders scheme
        returns all the possible paths'''
    rutes = []
    if not isinstance(dic, dict) or not dic: 
        return rute
    else:
        for key in dic:
            aux = folder_path(dic[key], rute = rute+key)
            if isinstance(aux,list):
                rutes += aux
            else:
                rutes.append(aux)
    return rutes

	
def create_rute(rute):
    ''' Creates rute if not exists.
        rute: path to create.'''
    if path.exists(rute):
        print(f'Already exists: "{rute}".')
    else:
        try:
            makedirs(rute)
            print(f'Created successfully: "{rute}".')
        except:
            print(f'Could not create: "{rute}".')		

def create_folders_scheme(scheme):
	''' Creates folders scheme.
		scheme: dict. Scheme of folders to create.
	'''
	for rute in folder_path(scheme):
		create_rute(rute)

		
#----------------------------------------------------------------------------------------#
#                            Preprocessing functions                                      #
#----------------------------------------------------------------------------------------#

def get_fov_mask(fundus_picture):
    '''
    Obtains the fov mask of the given fundus picture.
    '''

    # try to detect a cicle
    gray = cv2.cvtColor(fundus_picture, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)

    rows = gray_blur.shape[0]
    cols = gray_blur.shape[1]
    circles = cv2.HoughCircles(gray_blur,
                               method=cv2.HOUGH_GRADIENT,
                               dp=1,
                               minDist=cols//2,
                               param1=60,
                               param2=30,
                               minRadius=min(rows,cols)//4,
                               maxRadius=max(rows,cols)//2)

    if circles is not None and circles.shape[1] == 1:
        circles = np.uint16(np.around(circles))
        circle = circles[0,0]
        fov_mask = np.zeros((rows,cols),np.uint16)
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(fov_mask, center, radius, 255 , -1)

    else:
        sum_of_channels = np.sum(fundus_picture,axis=2)
        threshold_fondo = filters.threshold_otsu(sum_of_channels.astype(np.int64))
        fov_mask = sum_of_channels > threshold_fondo
        fov_mask = binary_fill_holes(fov_mask)

    return fov_mask
	

def crop_fov(fundus_picture, fov_mask):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    #get the fov mask of the picture
    fov_mask = get_fov_mask(fundus_picture)

    fov_mask = np.asarray(fov_mask, dtype=np.uint8)
    
    # get the coordinate of a bounding box around the fov mask
    coordinates = measure.regionprops(fov_mask)[0].bbox

    # Set to black every pixel outside the fov_mask
    fundus_picture[fov_mask<=0,...]=0

    # crop the image and return
    return fundus_picture[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:]

def set_black_background(fundus_picture):
    '''
    Set background to black if it is not already
    '''

    gray = cv2.cvtColor(fundus_picture, cv2.COLOR_BGR2GRAY)
    freq_intensity = np.bincount(gray.ravel().astype(np.int64))

    # If mode is 255, then it means the background is white
    if np.isin(255, np.where(freq_intensity == freq_intensity.max())[0]):
        index = np.where(gray == 255)
        imagen_modified = fundus_picture.copy()
        imagen_modified[index[0],index[1],...] = 0
    else:
        imagen_modified = fundus_picture.copy()

    return imagen_modified


def preprocess_and_save(pd_labels, save_to, image_size=[512, 512]):
    '''
    Preprocesses the images and saves the result.
    '''
    for image_dir in pd_labels.file_path:
        try:
            new_name = image_dir.with_suffix('.png').name
            im = cv2.imread(image_dir)
            im_new = set_black_background(im)
            fov_mask = get_fov_mask(im_new)
            im_new = crop_fov(im_new,fov_mask)
            im_new = cv2.resize(im_new, (512, 512))
            cv2.imwrite(str(save_to / new_name), im_new)  
        except Exception as e:
            print(f"Error processing {image_dir}: {e}")
    return


def parallel_arg(df,save_path):
    '''
    Splits the dataset into as many chunks as
    logical processors are in the cpu,
    or fewer if the dataset is small.
    '''
    chunks_count = min(mp.cpu_count(), len(df))
    chunk_size = len(df)//chunks_count
    chunk_args = []
    chunk_start = 0
    for i in range(chunks_count-1):
        chunk_args.append((df.iloc[chunk_start:chunk_size*(i+1)],save_path))
        chunk_start += chunk_size
    
    chunk_args.append((df.iloc[chunk_start:],save_path))

    return chunk_args

