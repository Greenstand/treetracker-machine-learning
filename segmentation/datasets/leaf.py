import os
import json
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
import torch

class RandomRescale(object):
    def __init__(self, min_size, max_size):
        self.output_size = np.random.randint(min_size, max_size)
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        mask = transform.resize(mask, (new_h, new_w))
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        mask = mask * [new_w / w, new_h / h]

        return {'image': img, 'mask': mask}
    
train_img_transform = transforms.Compose([
  #RandomCropAndPad(512),
  transforms.Resize((256, 256)),
  #transforms.RandomResizedCrop(size=(256, 256)),
  transforms.RandomHorizontalFlip(),
  #transforms.RandomRotation(degrees=(0, 360)),
  transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.0, hue=0.0),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_mask_transform = transforms.Compose([
    #RandomCropAndPadMask(512),
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    #transforms.RandomResizedCrop(size=(256, 256), interpolation=transforms.InterpolationMode.NEAREST),
   transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(degrees=(0, 360)),
    transforms.ToTensor(),
])

val_img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])
class LeafDataset(data.Dataset):
    def __init__(self, root, image_set='train', img_transform=None, mask_transform=None, ext=".jpg"):
        self.root = os.path.expanduser(root)
        print ("Root :" , self.root)
        print ("CWD: " , os.getcwd())
        print ("Children: ", os.listdir(os.path.join(os.getcwd(), "local_data", "splits")))
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
        sample = {"image": img,
                  "mask": mask}
        
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
