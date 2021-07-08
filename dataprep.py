import os
import glob
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
from imagecutter import imageCutter
import hashlib
import pickle

input_dir = r'C:\Users\Pascal\Pictures\dataset'
DIMENSION = (90, 90)

image_files = glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)


def getborders():
    border_x = random.randint(10, 15)
    border_l_r = round((random.random()), 1)
    border_l = round((border_x - 10) * border_l_r) + 5
    border_r = border_x - border_l

    border_y = random.randint(10, 15)
    border_u_r = round((random.random()), 1)
    border_u = round((border_y - 10) * border_u_r) + 5
    border_d = border_y - border_u

    border_x_final = (border_l, border_r)
    border_y_final = (border_u, border_d)
    return border_x_final, border_y_final


m = hashlib.sha256()
pickl_dict = {
    'input_arrays': [],
    'known_arrays': [],
    'target_arrays': [],
    'borders_x': [],
    'borders_y': [],
    'sample_ids': []
}
for image_file in tqdm(image_files, desc="Processing files"):
    with Image.open(image_file) as img:
        img_s = img.resize(DIMENSION, resample=Image.BILINEAR)
        image_array = np.array(img_s, dtype=np.uint8)

    border1, border2 = getborders()
    input_array, known_array, target_array = imageCutter(image_array, border1, border2)
    # Image.fromarray(np.uint8(input_array), 'L').show()
    # Image.fromarray(np.uint8(known_array), 'L').show()
    m.update(bytes(image_file, 'utf-8'))
    img_id = m.hexdigest()

    pickl_dict['input_arrays'].append(input_array)
    pickl_dict['known_arrays'].append(known_array)
    pickl_dict['target_arrays'].append(target_array)
    pickl_dict['borders_x'].append(border1)
    pickl_dict['borders_y'].append(border1)
    pickl_dict['sample_ids'].append(img_id)
    filename = os.path.join("pickl_files",
                            os.path.dirname(os.path.relpath(image_file, input_dir)).replace(os.path.sep, '_') + '.pkl')

with open(filename, 'wb') as handle:
    pickle.dump(pickl_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
