import h5py
import os
import numpy as np
from tqdm import tqdm
import torchvision

# root = '/home/ml/users/afeizi/datasets/hotels50k_restructured_5000'
root = '/home/ml/users/afeizi/datasets/hotels50k_v5_restructured'

# splits = ['train', 'val_seen', 'val_unseen', 'test_seen', 'test_unseen']
splits = ['train_small',
          'val1_small',
          'val2_small',
          'val3_small',
          'val4_small',
          'test1_small',
          'test2_small',
          'test3_small',
          'test4_small']
for s in splits:
    img_count = 0
    for i in torchvision.datasets.ImageFolder(root=os.path.join(root, s)).imgs:
        fn = os.path.split(i[0])[1]
        if fn[:2] != '._':
            img_count += 1

    data = h5py.File(os.path.join(root, f'hotels5k_{s}.h5'), 'w')
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    data.create_dataset(name='x', shape=(img_count,), dtype=dt)
    data.create_dataset(name='y', shape=(img_count, 1), dtype='int32')

    print(img_count)

    ix = 0
    for i in tqdm(torchvision.datasets.ImageFolder(root=os.path.join(root, s)).imgs):
        # i[1]: label, i[0]: root
        y = i[1]
        img_path = os.path.join(root, i[0])

        # fn needed for removing non-images starting with `._`
        fn = os.path.split(i[0])[1]
        if fn[:2] != '._':
            # add label and image to hdf5
            # with open(img_path, 'rb') as f:
            c_f = open(img_path, 'rb')
            img_bytes = c_f.read()
            c_f.close()
            data['x'][ix] = np.fromstring(img_bytes, dtype='uint8')
            data['y'][ix] = y

            ix += 1
    data.close()
