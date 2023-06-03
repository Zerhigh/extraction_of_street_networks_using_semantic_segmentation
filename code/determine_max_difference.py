import numpy as np
import os
from tqdm import tqdm
import skimage
import time
from itertools import combinations
import cv2

def read_8channel_from_path(image_path):
    """
        receives: file path
        returns: 8 channel image
    """
    image = skimage.io.imread(image_path, plugin='tifffile')
    return image

def normalize_2d(img):
    """
        receives: image
        returns: normalized image
    """
    norm = np.linalg.norm(img, 1)
    img = img/norm
    return img

def calculate_difference_between_channels(all):
    """
        receives: list of all images
        returns: sorted dict of maximum differences with channels as keys
        calculates the maximum difference between all channel combinations
    """

    all_imgs = np.empty((len(all), 325, 325, 8), dtype='uint16')

    for i, img in tqdm(enumerate(all)):
        if os.path.splitext(img)[1] == '.tif':
            img = read_8channel_from_path(img) # f'{img_paths}{img}'
            all_imgs[i, :, :, :] = img

    print(all_imgs.shape)
    """
    possible channel allocation -> unclear
    0   5   Coastal: 397–454 nm 
    1   3   Blue: 445–517 nm 
    2   2   Green: 507–586 nm 
    3   6   Yellow: 580–629 nm 
    4   1   Red: 626–696 nm
    5   4   Red Edge: 698–749 nm
    6   7   Near-IR1: 765–899 nm
    7   8   Near-IR2: 857–1039 nm
    """

    # calculate all channel combinations
    channel_triples = list(combinations(range(8), 3))
    print(channel_triples)

    # cslculate max diff for ach channel combination
    diff_values = dict()
    for tripel in channel_triples:
        name = f'{tripel[0]}_{tripel[1]}_{tripel[2]}'
        value = np.mean(np.abs(np.diff(np.array([   all_imgs[:, :, :, tripel[0]],
                                                    all_imgs[:, :, :, tripel[1]],
                                                    all_imgs[:, :, :, tripel[2]]]), axis=1)))
        diff_values[name] = value
        print(name, value)

    # sort dict after max diff
    sorted_dict = sorted(diff_values.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict)

    stop = time.time()
    print(stop-start)
    return sorted_dict

def write_channels_to_img(all_image_files, out_path, channels, rescale=False):
    """
        receives: all image files, saving path, selected channels, boolean wheter to rescaleor not
        returns: True
        reads in all images, and converts them from 8 channel to 3 selected channel images
    """

    for i, img in tqdm(enumerate(all_image_files)):
        img_name = os.path.splitext(img)[0].split('/')[-1]
        if os.path.splitext(img)[1] == '.tif':
            image = read_8channel_from_path(img)
            if rescale:
                save_image = image[:, :, channels] #(cv2.normalize(, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) * 255).astype('uint8')
            else:
                save_image = image[:, :, channels]
            cv2.imwrite(f'{out_path}/{img_name}.png', save_image)

    return True

start = time.time()
img_path_list = ['D:/SHollendonner/data_3/AOI_2_Vegas/', 'D:/SHollendonner/data_3/AOI_4_Shanghai/', 'D:/SHollendonner/data_3/AOI_5_Khartoum/', 'D:/SHollendonner/data_3/AOI_3_Paris/']

all_names_comp = list()
for l in img_path_list:
    for file in os.listdir(l+'MS/'):
        if 'aux' not in file:
            all_names_comp.append(l+'MS/'+file)

sort_dict = calculate_difference_between_channels(all_names_comp)
print(sort_dict)

# decided on channel combination (2, 5, 7), to check in qgis select channel (3, 6, 8)
out_path = 'D:/SHollendonner/multispectral/images_257'
out_path8BIT = 'D:/SHollendonner/multispectral/images8Bit/'
#all_names8Bit = [out_path8BIT+img for img in os.listdir(out_path8BIT)]
#write_channels_to_img(all_image_files=all_names8Bit, out_path=out_path, channels=[2, 5, 7], rescale=True)
