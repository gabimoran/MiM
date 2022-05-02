from os import path, listdir, makedirs
import numpy as np
from matplotlib import pyplot as plt

from skimage import filters, measure, img_as_ubyte
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import rescale
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import multiprocessing as mp 
from multiprocessing import Pool
		
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

    # sum the R, G, B channels to form a single image
    sum_of_channels = np.asarray(np.sum(fundus_picture,axis=2), dtype=np.uint8)
    # threshold the image using Otsu
    fov_mask = sum_of_channels > filters.threshold_otsu(sum_of_channels)
    # fill holes in the approximate FOV mask
    fov_mask = np.asarray(binary_fill_holes(fov_mask), dtype=np.uint8)

    return fov_mask
	

def crop_fov(fundus_picture):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    #get the fov mask of the picture
    fov_mask = get_fov_mask(fundus_picture)

    # get the coordinate of a bounding box around the fov mask
    coordinates = measure.regionprops(fov_mask)[0].bbox

    # crop the image and return
    return fundus_picture[coordinates[0]:coordinates[2],coordinates[1]:coordinates[3],:]

def crop_fov_pill(fundus_picture):
    '''
    Extract an approximate FOV mask, and crop the picture around it
    '''

    #get the fov mask of the picture
    fov_mask = get_fov_mask(fundus_picture)

    # get the coordinate of a bounding box around the fov mask
    coordinates = measure.regionprops(fov_mask)[0].bbox

    # crop the image and return
    return fundus_picture.crop((coordinates[1],coordinates[0],coordinates[3],coordinates[2]))

def downsize_image(fundus_picture, image_size=[512, 512]):
    '''
    Downsize the input image to the target resolution
    '''

    # get the proper size
    if fundus_picture.size[0] <= fundus_picture.size[1]:
        factor = image_size[0] / fundus_picture.size[0]
    else:
        factor = image_size[0] / fundus_picture.size[1]
    
    target_size = (round(fundus_picture.size[0] * factor), round(fundus_picture.size[1] * factor))

    # apply the transformation
    resized_fundus_picture = fundus_picture.resize(size=target_size)

    return resized_fundus_picture


def preprocess_and_save(pd_labels, save_to, cut_text=False ):
    '''
    Preprocesses the images and saves the result.
    At the end shows the last image before and after preproccessing.
    '''

    for image_dir in pd_labels.file_path:
        new_name = path.split(image_dir.with_suffix('.png'))[-1]
        im = Image.open(image_dir)
        if cut_text:
            left = 200
            right = 1100
            top = 30
            bottom = im.getbbox()[3]
            im_cropped = im.crop((left,top,right,bottom)) # recorta la parte de la imagen con inscripciones.
            im_fov = crop_fov_pill(im_cropped)
        else:
            im_fov = crop_fov_pill(im)

        im_fov = downsize_image(im_fov)
        
        im_fov.save(save_to/new_name)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(im)

    ax[1].imshow(im_fov,cmap=plt.cm.gray) 
    plt.show()


def parallel_arg(df,save_path):
    '''
    Splits the dataset into as many chunks as
    logical processors are in the cpu.
    '''
    chunks_count = mp.cpu_count()
    chunk_size = df.shape[0]//chunks_count
    chunk_args = []
    chunk_start = 0
    for i in range(chunks_count-1):
        chunk_args.append((df.iloc[chunk_start:chunk_size*(i+1)],save_path))
        chunk_start += chunk_size
    
    chunk_args.append((df.iloc[chunk_start:],save_path))

    return chunk_args