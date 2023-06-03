import matplotlib.pyplot as plt
import os
import time
import numpy as np
import datetime
import random
import pickle
# change cv2s limitation on image size
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
from tqdm import tqdm
import shutil
from PIL import Image
from osgeo import gdal, ogr, osr

def square_image(image):
    squared_image = image[:min(image.shape[:2]), :min(image.shape[:2]), :]
    return squared_image

def tile_image(image, tile_size, pixel_size, overlap):
    """
        receives: uploadable image should be square starting in the lower left corner it will be squared, requiered
            size in m, given size of a pixel, overlap of two adjacent pictures in m
        returns: list of squaring image indices
    """

    # square image if necessary
    if image.shape[0] != image.shape[1]:
        image = square_image(image)
    increment = tile_size * pixel_size
    no_images = int(np.floor(image.shape[0]/increment))
    ret_list = list()
    for i in range(no_images-1):
        for j in range(no_images-1):
            ret_list.append(image[i*increment : (i+1)*increment, j*increment : (j+1)*increment, :])

    return ret_list

def tile_image_w_overlap(image, parameter_dict, save_name, tiled_images, do_augment=True):
    """
        receives: uploadable image should be square starting in the lower left corner it will be squared, dict with
            parameters, saving path, list of rference images, boolean for augmentation
        returns: dict mapping the split images
        tiles all images and saves onto disk
    """

    # define redundancy
    s = time.time()
    tile_size = parameter_dict['tile_size']
    pixel_size = parameter_dict['pixel_size']
    overlap = parameter_dict['overlap']
    save = parameter_dict['save']
    save_path = parameter_dict['save_path']
    tiling_mask = parameter_dict['tiling_mask']
    overlap_indices = parameter_dict['overlap_indices']

    # square image if necessary
    if image.shape[0] != image.shape[1]:
        image = square_image(image)

    no_images = int(len(overlap_indices))
    len_id = len(str(no_images))
    ret_dict = dict()

    if parameter_dict['MS']:
        num_channels = 8
    else:
        num_channels = 3

    for i in range(no_images):
        # create first index or naming convention
        first_ind = list(str(i))
        while len(first_ind) <= len_id:
            first_ind.insert(0, '0')
        fi = ''.join(first_ind)
        for j in range(no_images):
            # create second index or naming convention
            second_ind = list(str(j))
            while len(second_ind) <= len_id:
                second_ind.insert(0, '0')
            si = ''.join(second_ind)

            add_image = image[overlap_indices[i][0]:overlap_indices[i][1], overlap_indices[j][0]:overlap_indices[j][1], :]

            # tiling into 256 can lead to rounding error, solved by assigning and cropping a larger base image
            if add_image.shape != (parameter_dict['shape'], parameter_dict['shape'], num_channels):
                base_img = np.zeros((parameter_dict['shape'], parameter_dict['shape'], num_channels))
                base_img[:add_image.shape[0], :add_image.shape[1], :] += add_image
                add_image = base_img

            if save:

                if (not tiling_mask and len(np.unique(add_image)) > 1) or (tiling_mask and f'{save_name.split(".")[0]}_{fi}_{si}.png' in tiled_images): #SN3_roads_train_
                    # apply augmentations
                    if do_augment:
                        horizontal_flip = add_image[:, ::-1, :]
                        vertical_flip = add_image[::-1, :, :]
                        horizontal_vertical_flip = add_image[::-1, ::-1, :]
                        cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}_hn_rot0.png', horizontal_flip)
                        cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}_vn_rot0.png', vertical_flip)
                        cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}_hv_rot0.png', horizontal_vertical_flip)

                        # rotate
                        for rot in range(3):
                            rot_img = np.rot90(add_image, rot+1)
                            cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}_nn_rot{rot+1}.png', rot_img)

                        #print(f'saved {save_path}/{save_name.split(".")[0]}_{fi}_{si}.tif AND augmented versions')

                    if not parameter_dict['split']:
                        if parameter_dict['MS']:
                            with open(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}.pickle', "wb") as f:
                                pickle.dump(add_image, f)
                        else:
                            cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}.png', add_image)

                    # split image
                    if parameter_dict['split'] and not tiling_mask:
                        if random.random() <= 0.8:
                            cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}_nn_rot0.png', add_image)
                        else:
                            cv2.imwrite(f'{save_path}_test/{save_name.split(".")[0]}_{fi}_{si}_nn_rot0.png', add_image)

                    # split mask
                    elif parameter_dict['split'] and tiling_mask:
                        if f'{save_name.split(".")[0]}_{fi}_{si}_nn_rot0.png' in os.listdir(f'{"/".join(save_path.split("/")[:-1])}/images/'):
                            cv2.imwrite(f'{save_path}/{save_name.split(".")[0]}_{fi}_{si}_nn_rot0.png', add_image)
                        else:
                            cv2.imwrite(f'{save_path}_test/{save_name.split(".")[0]}_{fi}_{si}_nn_rot0.png', add_image)

            # check and dont include monocolor areas
            if len(np.unique(add_image)) > 5:
                ret_dict[f'img_{fi}_{si}'] = add_image
    e = time.time()
    return ret_dict

def save_images(path, folder_name, image_dict, params):
    """
        receives: base path, folder name, image dict conatining images, image params
        returns: True
        saves images with parameter
    """

    s = time.time()
    os.mkdir(f'{path}tiled_images/{folder_name}')
    for key, image in image_dict.items():
        cv2.imwrite(f'{path}tiled_images/{folder_name}/{key}.tif', image)
    with open(f'{path}tiled_images/{folder_name}/README.txt', 'w') as file:
        file.write(f'tiling image {folder_name} on {datetime.datetime.now().isoformat()} \n')
        for key, value in params.items():
            file.write('%s:%s\n' % (key, value))

    e = time.time()
    print(f'saving {len(image_dict)} images took {e - s} [s]')
    return True

def determine_overlap(img_size, wish_size):
    """
        receives: image size to split, size image is split into
        returns: list of tuples describing the indices to split an image along
        calculates indices on whichan image has to be split
    """

    num_pics = int(np.ceil(img_size/wish_size))
    applied_step = int((num_pics * wish_size - img_size) / (num_pics - 1))
    overlap_indices = [(i*(wish_size-applied_step), (i+1)*wish_size - i*applied_step) for i in range(num_pics)]
    print(overlap_indices)

    return overlap_indices

def apply_tiling(source_name, folder_name, wish_size):
    """
        receives: paht, folder path name, size to tile into
        returns: True
        Tiles images and masks into the wish size with an overlap
    """

    # define parameter
    params_masks = {'tile_size': 1300,
              'pixel_size': 0.3,
              'overlap': 0,
              'save': True,
              'save_path': f'{source_name}/{folder_name}/rehashed_ones',
              'tiling_mask': True,
              'overlap_indices': determine_overlap(1300, wish_size),
              'split': False,
              'shape': wish_size,
              'MS': False}

    # define parameter
    params_images = {'tile_size': 1300,
              'pixel_size': 0.3,
              'overlap': 0,
              'save': True,
              'save_path': f'{source_name}/{folder_name}/images',
              'tiling_mask': False,
              'overlap_indices': determine_overlap(1300, wish_size),
              'split': False,
              'shape': wish_size,
              'MS': False}

    tiled_images = []

    # create folders
    if 'images' not in os.listdir(f'{source_name}{folder_name}'):
        os.mkdir(f'{source_name}{folder_name}/images')
    if 'masks2m' not in os.listdir(f'{source_name}{folder_name}'):
        os.mkdir(f'{source_name}{folder_name}/masks2m')
    if 'rehashed' not in os.listdir(f'{source_name}{folder_name}'):
        os.mkdir(f'{source_name}{folder_name}/rehashed')

    # tile images
    for img_name in tqdm(os.listdir(f'{source_name}/images')):
        if params_images['MS']:
            # gdal array is read in as (8, 1300, 1300), but is needed as a (1300, 1300, 8). transposin switches these axes
            img = np.transpose(gdal.Open(f'{source_name}/images/{img_name}').ReadAsArray(), (1, 2, 0))
        else:
            img = cv2.imread(f'{source_name}/images/{img_name}')
        res1 = tile_image_w_overlap(img, params_images, img_name, tiled_images, do_augment=False)

    # access tiled images to check if corresponfing mask is available
    tiled_images = os.listdir(f'{source_name}/{folder_name}/images/') #+ os.listdir(f'{source_name}/{folder_name}/images_test/')
    img_errors = []

    # tile masks
    for img_name in tqdm(os.listdir(f'{source_name}/rehashed')):
        try:
            img = cv2.imread(f'{source_name}/rehashed/{img_name}')
            res1 = tile_image_w_overlap(img, params_masks, img_name, tiled_images, do_augment=False)
        except:
            img_errors.append(img_name)
            print("stupid image error")

    return True

def masks_to_single_channel(path):
    """
        receives: path
        returns: True
        converts 3 channel masks into single channel masks
    """

    for img in tqdm(os.listdir(path+'/rehashed/')):
        re_img = cv2.imread(f'{path}/rehashed/{img}')
        sh = re_img.shape
        out = np.reshape(re_img[:, :, 1], (sh[0], sh[1], 1))
        cv2.imwrite(f'{path}/rehashed_one/{img}', out)

    return  True

def split_after_augmentation(base_path, folder_path):
    """
        receives: path, path to folder
        returns: True
        splits images manually into training and validation data
    """

    nec_folders = ['images', 'images_test', 'masks2m', 'masks2m_test', 'rehashed', 'rehashed_test']
    for folder in os.listdir(base_path+folder_path):
        assert folder in nec_folders
    # continue
    for i, m, r in tqdm(zip(os.listdir(base_path+folder_path+'/images'), os.listdir(base_path+folder_path+'/masks2m'), os.listdir(base_path+folder_path+'/rehashed'))):
        if random.random() <= 0.2:
            shutil.move(base_path+folder_path+'/images/'+i, base_path+folder_path+'/images_test')
            shutil.move(base_path + folder_path + '/masks2m/'+m, base_path + folder_path + '/masks2m_test')
            shutil.move(base_path + folder_path + '/rehashed/'+r, base_path + folder_path + '/rehashed_test')
            print('moved')

    return True

def rename_files(folder_dir):
    """
        receives: folder path
        returns: True
        Renames images or masks into a png
    """

    if 'images' in os.listdir(folder_dir) or 'rehashed' in os.listdir(folder_dir):
        folders = os.listdir(folder_dir)
        print(folders)
        for folder in tqdm(folders):
            print(folder)
            if folder in ['images', 'rehashed']:
                for image in os.listdir(f'{folder_dir}/{folder}'):
                    os.rename(f'{folder_dir}/{folder}/{image}', f'{folder_dir}/{folder}/{image.split(".")[0]}.png')

    return True

def add_buffer(image_path, save_path, buffer_size, base_size):
    """
        receives: path, path to save, applied buffer size, image size
        returns: True
        buffer size will be added to each edge of the image eg. buffer_size = 22 -> image size = base_size + (2 * buffer_size) = 1344
    """

    for img in tqdm(os.listdir(image_path)):
        base_img = np.zeros((base_size + 2*buffer_size, base_size + 2*buffer_size, 3))

        image = np.asarray(Image.open(f"{image_path}{img}"))
        base_img[buffer_size:base_size+buffer_size, buffer_size:base_size+buffer_size, :] = image
        cv2.imwrite(f'{save_path}{os.path.splitext(img)[0]}.png', base_img)

    return True

start = time.time()

base_source = 'D:/SHollendonner/data_5/AOI_8_Mumbai/'#'D:/SHollendonner/MS/'#'C:/Users/shollend/bachelor/test_data/train/'
folder_name = 'tiled256/'#'tiled512_overlap_augment_png'
big_img_path = 'D:/SHollendonner/images/'
big_img_save_path = 'D:/SHollendonner/images_buffer/'

# apply_tiling(base_source, folder_name, wish_size=256)

# determine_overlap(1300, 256)

# masks_to_single_channel(f'{base_source}{folder_name}')

# split_after_augmentation(base_source, folder_name)

# rename_files(base_source+folder_name)

# add buffer to big images

# add_buffer(big_img_path, big_img_save_path, 22, 1300)

stop = time.time()
print(f'time of script running [s]: {round(stop-start, 5)}')


