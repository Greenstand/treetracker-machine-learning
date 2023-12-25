import os
import json
import numpy as np
from PIL import Image
import torch.utils.data as data
import cv2
import torch

class LeafDataset(data.Dataset):
    def __init__(self, root, image_set='train', img_transform=None, mask_transform=None, ext=".jpg"):
        self.root = os.path.expanduser(root)
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_set = image_set        # Adjust paths
        image_dir = os.path.join(self.root, 'samples')
        mask_dir = os.path.join(self.root, 'binary_masks')
        split_fpath = os.path.join(self.root, 'splits', f'{self.image_set}.txt')        
        with open(split_fpath, 'r') as f:
            file_names = [x.strip().lower() for x in f.readlines()]        
            self.images = [os.path.join(image_dir, fname + ext) for fname in file_names]
            self.masks = [os.path.join(mask_dir, fname + '_mask_' + ext) for fname in file_names]

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        mask_array = np.array(mask)
        mask_array = (mask_array > 128).astype(np.uint8) # Binarize to 0s and 1s

        mask_array = mask_array * 255
        mask = Image.fromarray(mask_array.astype(np.uint8))
        
        if self.img_transform is not None:
          img = self.img_transform(img)
        
        if self.mask_transform is not None:
          mask = self.mask_transform(mask)

        mask = torch.squeeze(mask, 0)

        return img, mask

    @staticmethod
    def decode_target(mask):
        """Decode binary mask to RGB image"""
        leaf_color = [255, 0, 255]
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[mask == 1] = leaf_color

        return Image.fromarray(rgb_mask)


    def __len__(self):
        return len(self.images)
