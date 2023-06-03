from PIL import Image
import os
from tqdm import tqdm

file_path = 'D:/SHollendonner/multispectral/channels_257/tiled512/' #D:\SHollendonner\multispectral\channels_257\tiled512

def apply_rotations(file_path):
    """
        receives: file path
        returns: True
        rotates images and masks for 90 degrees three times
    """

    if not os.path.exists(f'{file_path}/rotated/'):
        os.mkdir(f'{file_path}/rotated/')
    if not os.path.exists(f'{file_path}/rotated/images/'):
        os.mkdir(f'{file_path}/rotated/images/')
    if not os.path.exists(f'{file_path}/rotated/rehashed_ones/'):
        os.mkdir(f'{file_path}/rotated/rehashed_ones/')

    for img_ in tqdm(os.listdir(f'{file_path}/images/')):
        img = Image.open(f'{file_path}/images/{img_}')
        for i in range(4):
            img_save = img.rotate(90*(i))
            img_save.save(f'{file_path}/rotated/images/{os.path.splitext(img_)[0]}_rot{90*(i)}.png')

    for img_ in tqdm(os.listdir(f'{file_path}/rehashed_ones/')):
        img = Image.open(f'{file_path}/rehashed_ones/{img_}')
        for i in range(4):
            img_save = img.rotate(90*(i))
            img_save.save(f'{file_path}/rotated/rehashed_ones/{os.path.splitext(img_)[0]}_rot{90*(i)}.png')
    return True

# apply_rotations(file_path)

