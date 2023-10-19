import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import nibabel as nib
import cv2
import tarfile
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

IMG_SIZE=128
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5
VOLUME_SLICES = 128
VOLUME_START_AT = 22 # first slice of volume that we will include

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

TRAIN_DATASET_PATH = '/home/sucheng/Pytorch-UNet/data/brain_images'
VALIDATION_DATASET_PATH = TRAIN_DATASET_PATH + '/val_brain_images'
TEST_DATASET_PATH = TRAIN_DATASET_PATH  + './test_brain_images'


class BratsDataset(Dataset):
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        # 初始化Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch '
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = torch.zeros((self.batch_size * VOLUME_SLICES, 2, *self.dim, self.n_channels))  # 使用torch.Tensor代替NumPy数组
        y = torch.zeros((self.batch_size * VOLUME_SLICES, 4, IMG_SIZE, IMG_SIZE))  # 只有一个通道的标签图像
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii.gz')
            flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_t1ce.nii.gz')
            ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii.gz')
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):
                flair_slice = torch.from_numpy(flair[:, :, j + VOLUME_START_AT]).unsqueeze(0).unsqueeze(0)  # 将二维数据扩展为三维
                ce_slice = torch.from_numpy(ce[:, :, j + VOLUME_START_AT]).unsqueeze(0).unsqueeze(0)  # 将二维数据扩展为三维

            # 使用torch.nn.functional.interpolate()进行插值，将图像大小调整为(IMG_SIZE, IMG_SIZE)
                flair_interpolated = F.interpolate(flair_slice, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                ce_interpolated = F.interpolate(ce_slice, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                X[j + (VOLUME_SLICES * c), 0, :, :] = flair_interpolated
                X[j + (VOLUME_SLICES * c), 1, :, :] = ce_interpolated

                y[j + (VOLUME_SLICES * c), :, :] = cv2.resize(seg[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
                # X[j+(VOLUME_SLICES*c), 0,:,:] = F.interpolate(flair_slice, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
                # X[j+(VOLUME_SLICES*c), 1,:,:] = F.interpolate(ce_slice, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
                # y[j +VOLUME_SLICES*c,:,:] = cv2.resize(seg[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

        # Generate masks(One-Hot encoding)
        y[y==4] = 3
        y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=4).permute(0, 3, 1, 2).float()  # 进行One-Hot编码
        #Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        #Y = np.array(Y).reshape(1,128,128,128)
        return X / X.max(), y  # 返回归一化后的输入图像和One-Hot编码后的标签

train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

train_and_test_ids = pathListIntoIds(train_and_val_directories)
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)

training_generator = BratsDataset(train_ids)
valid_generator = BratsDataset(val_ids)
test_generator = BratsDataset(test_ids)
