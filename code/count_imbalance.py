import os
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def read_image(source):
    image = np.asarray(Image.open(source)).flatten()
    return image

base = glob(f'D:/SHollendonner/not_tiled/rehashed/*.png')

def count_imbalance(path):
    """
        receives: file path
        returns: True
        calculates thepixel imbalance of fore and background and plots a historgam of its distribution
    """

    street = 0
    background = 0
    imgs = len(path)
    all_means_str = list()
    px_p_img = 1690000

    for mask in tqdm(path):
        img = read_image(mask)
        str = np.count_nonzero(img == 1)
        back = np.count_nonzero(img == 0)
        if int(str + back) != int(px_p_img):
            print('error')
        street += str
        background += back
        all_means_str.append(100*(str/px_p_img))

    prc_street = street/(px_p_img*imgs)
    prc_background = background/(px_p_img*imgs)


    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    ax.grid(which='major', linestyle='--', alpha=0.5)
    ax.hist(np.array(all_means_str), bins=100, density=True, color='tab:olive')
    ax.set_xlabel('Percentage of street labeled pixels', fontsize=20)
    ax.set_ylabel('Frequency', fontsize=20)
    #ax.set_title('Histogram showing the imbalance of percentage of street labeled pixels in the SpaceNet dataset')
    # Add a legend
    #tab:olive
    #ax.legend(patches, ['Group 1', 'Group 2', 'Group 3', 'Group 4'])

    # Customize the appearance
    path2 = 'D:/SHollendonner/graphics/'
    fig.tight_layout()
    plt.savefig(f'{path2}/class_imbalance.png')
    plt.show()

    print(prc_street, prc_background, prc_street+prc_background)
    print(street, background, np.mean(np.array(all_means_str)))

    return True

count_imbalance(base)