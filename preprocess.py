import os
import random
import monai.transforms as transforms
from monai.data import DataLoader, CacheDataset
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

def pre_crop_transforms() -> transforms:
    return transforms.Compose([
        transforms.LoadImaged(keys=['ct', 'mri', 'mask'], reader='NibabelReader'),
        transforms.ScaleIntensityd(keys=['ct', 'mri']),
        # transforms.ScaleIntensityRanged(keys=['ct'], a_min=-1000, a_max=1800, b_min=0.0, b_max=1.0, clip=True,),
        # transforms.ScaleIntensityRanged(keys=['mri'], a_min=0, a_max=1400, b_min=0.0, b_max=1.0, clip=True,),

        transforms.ToTensord(keys=['ct', 'mri', 'mask']),
    ])


def post_crop_transforms(img_size=(128,128)) -> transforms:
    return transforms.Compose([
        transforms.EnsureChannelFirstd(keys=['ct', 'mri', 'mask']),
        # img size (168, 168)
        transforms.Resized(keys=['ct', 'mri', 'mask'], spatial_size=img_size, mode='nearest'),
    ])


def crop_by_mask(data):
    res = []
    prog_bar_crop = tqdm(data)
    prog_bar_crop.set_description(f'Cropping images')
    for datapoint in prog_bar_crop:
        res.append(crop_each_by_mask(datapoint))
    return res


def crop_each_by_mask(data):
    maxes = {
        'y': 0,
        'x': 0,
        'all': 0,
    }
    largest_mask = {
        'y': None,
        'x': None,
        'all': None,
    }
    largest_mri = {
        'y': None,
        'x': None,
        'all': None,        
    }
    largest_ct = {
        'y': None,
        'x': None,
        'all': None,        
    }
    
    for slc in range(data['mask'].shape[-1]):
        if slc < 10 or slc > data['mask'].shape[-1] - 10:
            continue
        
        # calculate bounds of a bounding box
        xmax = np.max(np.argwhere(data['mask'][..., slc])[..., 0])
        xmin = np.min(np.argwhere(data['mask'][..., slc])[..., 0])
        ymax = np.max(np.argwhere(data['mask'][..., slc])[..., 1])
        ymin = np.min(np.argwhere(data['mask'][..., slc])[..., 1])

        # first, crop images based on calculated bounding box
        cropped_mask = data['mask'][xmin:xmax, ymin:ymax, slc]
        cropped_mri = data['mri'][xmin:xmax, ymin:ymax, slc]
        cropped_ct = data['ct'][xmin:xmax, ymin:ymax, slc]

        # compare and update maxes
        # max shape at X dim
        # if cropped_mask.shape[0] > maxes['x']:
        #     maxes['x'] = cropped_mask.shape[0]
        #     largest_mask['x'] = cropped_mask
        #     largest_mri['x'] = cropped_mri * cropped_mask
        #     largest_ct['x'] = cropped_ct * cropped_mask

        # max shape at Y dim
        if cropped_mask.shape[1] > maxes['y']:
            maxes['y'] = cropped_mask.shape[1]
            largest_mask['y'] = cropped_mask
            largest_mri['y'] = cropped_mri * cropped_mask
            largest_ct['y'] = cropped_ct * cropped_mask
            
            # do not take slice largest in x axis, but take one slice below the largest in y axis
            largest_mask['x'] = data['mask'][xmin:xmax, ymin:ymax, slc - 1]
            largest_mri['x'] = data['mri'][xmin:xmax, ymin:ymax, slc - 1] * largest_mask['x']
            largest_ct['x'] = data['ct'][xmin:xmax, ymin:ymax, slc - 1] * largest_mask['x']
            
            # do not take slice largest in all axis, but take one slice above the largest in y axis
            largest_mask['all'] = data['mask'][xmin:xmax, ymin:ymax, slc + 1]
            largest_mri['all'] = data['mri'][xmin:xmax, ymin:ymax, slc + 1] * largest_mask['all']
            largest_ct['all'] = data['ct'][xmin:xmax, ymin:ymax, slc + 1] * largest_mask['all']

        # max X + Y dims
        # if cropped_mask.shape[0] + cropped_mask.shape[1] > maxes['all']:
        #     maxes['all'] = cropped_mask.shape[0] + cropped_mask.shape[1]
        #     largest_mask['all'] = cropped_mask
        #     largest_mri['all'] = cropped_mri * cropped_mask
        #     largest_ct['all'] = cropped_ct * cropped_mask

    # return only the 3 tracked maxes
    return {
        'mri': [largest_mri['x'], largest_mri['y'], largest_mri['all']],
        'ct': [largest_ct['x'], largest_ct['y'], largest_ct['all']],
        'mask': [largest_mask['x'], largest_mask['y'], largest_mask['all']]
    }


def create_loaders(
        train_images: List[Dict],
        val_images: List[Dict],
        batch_size: int) -> Tuple[DataLoader, DataLoader]:

    tr = DataLoader(
        dataset=CacheDataset(data=train_images),
        batch_size=batch_size
    )
    val = DataLoader(
        dataset=CacheDataset(data=val_images),
        batch_size=batch_size
    )

    return tr, val


def preprocess_data(
        train_images: List[Dict],
        val_images: List[Dict],
        img_size) -> Tuple[List[Dict], List[Dict]]:
    
    # firstly, apply pre-crop transforms
    train_data = pre_crop_transforms()(train_images)
    val_data = pre_crop_transforms()(val_images)

    # crop the images based on the mask
    train_cropped = crop_by_mask(train_data)
    val_cropped = crop_by_mask(val_data)

    # collect the cropped data to a suitable form (list of dicts)
    preliminary_train_data = []
    for datapoint in train_cropped:
        preliminary_train_data.append({'mri': datapoint['mri'][0], 'ct': datapoint['ct'][0], 'mask': datapoint['mask'][0]})
        preliminary_train_data.append({'mri': datapoint['mri'][1], 'ct': datapoint['ct'][1], 'mask': datapoint['mask'][1]})
        preliminary_train_data.append({'mri': datapoint['mri'][2], 'ct': datapoint['ct'][2], 'mask': datapoint['mask'][2]})

    preliminary_val_data = []
    for datapoint in val_cropped:
        preliminary_val_data.append({'mri': datapoint['mri'][0], 'ct': datapoint['ct'][0], 'mask': datapoint['mask'][0]})
        preliminary_val_data.append({'mri': datapoint['mri'][1], 'ct': datapoint['ct'][1], 'mask': datapoint['mask'][1]})
        preliminary_val_data.append({'mri': datapoint['mri'][2], 'ct': datapoint['ct'][2], 'mask': datapoint['mask'][2]})
    
    # lastly, apply the post-crop transforms
    train_data = post_crop_transforms(img_size=img_size)(preliminary_train_data)
    val_data = post_crop_transforms(img_size=img_size)(preliminary_val_data)

    return train_data, val_data


def get_data_paths(
        data_dir: str,
        slice_ratio: float = 0.75,
        shuffle_seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict]]:
    train_images = []
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            ct_path = os.path.join(root, dir_name, "ct.nii.gz")
            mri_path = os.path.join(root, dir_name, "mr.nii.gz")
            mask_path = os.path.join(root, dir_name, "mask.nii.gz")
            if os.path.exists(ct_path) and os.path.exists(mri_path):
                train_images.append({'ct': ct_path, 'mri': mri_path, 'mask': mask_path})

    # shuffle the images based on the seed if the seed was given
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(train_images)
    slice_idx = int(len(train_images) * slice_ratio)
    val_images = train_images[slice_idx:]
    train_images = train_images[:slice_idx]

    return train_images, val_images


# tr_imgs, val_imgs = get_data_paths('/Users/simonukus/Documents/000FIIT/5_semester/BP_1/SynthRad/data/brain')
# train_loader, val_loader = create_loaders(tr_imgs, val_imgs, 1, (128, 128))
