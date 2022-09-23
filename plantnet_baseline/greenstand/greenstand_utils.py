# Imports
import numpy as np
import pandas as pd
import itertools
import time
from pathlib import Path
import s3_api
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from copy import deepcopy
import utils as plantnet_utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop


# Functions
def sync_split_get_dataloaders(args, skip_sync=False):
    """
    For ease of use: Performs all actions to simply get the DataLoaders needed!
    """
    # Ensure local files are present
    if not skip_sync:
        change = get_all_local_files(args)
    metadata = None
    change = True ## SET THIS TRUE, BUG DEALING WITH RANDOM CLASS NAMES

    # If there was a change in local files, recreate train/val/test split
    if change:
        metadata = perform_split(args)

    # Get metadata into memory
    if not metadata:
        metadata = load_metadata(args)

    # Get DataLoaders
    train, val, test, dataset_attributes = get_data_loaders(args, metadata)
    
    # Visualize if necessary
    if args['visualize'] == 'y':
        visualize_imbalance(dataset_attributes)
    
    return train, val, test, dataset_attributes


def get_all_local_files(args):
    """
    Pulls down all missing files from an S3 bucket and a list of prefixes
    """
    print("Checking to make sure all local files are present...")
    change = False
    for prefix in args['prefixes']:
        c = s3_api.get_missing_local_files(args['bucket'], prefix, args['local_path'], args['sub_dir_limit'])
        change = max(change, c)
    return change


def load_datasets(args, metadata=None):
    """
    Loads dataset(s). 
    If metadata is provided, loads a dataset for train, val, test separately
    If not, loads all data into one object.
    """
    print("Loading datasets...")
    # Define pathing and transforms
    paths = [args['local_path'] + "/" + p for p in os.listdir(args['local_path'])]
    transform = transforms.Compose(
        [MaxCenterCrop(), transforms.Resize(args['size_image']), transforms.ToTensor()])
    
    if not metadata:
        return GreenstandLabelledImageDataset(paths, transform=transform)
    else:
        train_set = GreenstandLabelledImageDataset(paths, transform=transform, images=metadata['train'])
        val_set = GreenstandLabelledImageDataset(paths, transform=transform, images=metadata['val'])
        test_set = GreenstandLabelledImageDataset(paths, transform=transform, images=metadata['test'])
        return train_set, val_set, test_set
        
        
def perform_split(args):
    """
    Ensure that we get a fair split per class of data. Generates a metadata.json file.
    """
    print("Creating metadata file with pre-determined train, val, test splits...")
    # Load up the dataset
    dataset = load_datasets(args, metadata=None)
    
    # Get num instances of each class
    num_instances_per_class = dataset.get_num_instances_per_class()
    total_num_instances = {}
    
    # Create splits per class
    metadata = {'invalid': [], 'total_instances_per_class' : total_num_instances}
    for c in num_instances_per_class:
        # Ensure we have enough instances per class to train, val, test
        num_instances = num_instances_per_class[c]
        class_name = dataset.classes[c]
        if num_instances < 3:
            metadata['invalid'].append(class_name)
            print(f"Will not include class {class_name} due to not having enough data for train/val/test split...")
            continue
            
        # Get data
        images = dataset.images[np.where(dataset.images[:, 1]==class_name)]
        total_num_instances[class_name] = len(images)
        
        # Calculate splits
        train_end = int(args['train_test_split'] * args['train_val_split']* num_instances)
        val_end = int(args['train_test_split'] * num_instances)
        test_end = num_instances
        
        # Alter indices - no val data
        if val_end - train_end < 1:
            train_end -= 1
            
        # Alter indices - no test data
        if test_end - val_end < 1:
            train_end -= 1
            val_end -= 1
            
        # Create splits
        train = images[0:train_end]
        val = images[train_end:val_end]
        test = images[val_end:]
        
        # Ensure they all have data
        assert train.shape[0] > 0
        assert val.shape[0] > 0
        assert test.shape[0] > 0   
        
        # Add to metadata
        if 'train' in metadata:
            metadata['train'] = np.concatenate((metadata['train'], train))
        else:
            metadata['train'] = train
        if 'val' in metadata:
            metadata['val'] = np.concatenate((metadata['val'], val))
        else:
            metadata['val'] = val
        if 'test' in metadata:
            metadata['test'] = np.concatenate((metadata['test'], test))
        else:
            metadata['test'] = test
          
    # Write metadata.json
    metadata_f = deepcopy(metadata)
    metadata_f['train'] = metadata_f['train'].tolist()
    metadata_f['val'] = metadata_f['val'].tolist()
    metadata_f['test'] = metadata_f['test'].tolist()
    print(f"Writing {args['metadata_file']}...")
    with open(args['metadata_file'], 'w') as file:     
        file.write(json.dumps(metadata_f))
    
    return metadata  
        
    
def load_metadata(args):
    """
    Loads the metadata file into memory
    """
    if os.path.exists(args['metadata_file']):
        print(f"Loading {args['metadata_file']}...")
        with open(args['metadata_file']) as file:
            metadata = json.loads(file.read())
            metadata['train'] = np.array(metadata['train'])
            metadata['val'] = np.array(metadata['val'])
            metadata['test'] = np.array(metadata['test'])
    else:
        print(f"No {args['metadata_file']} file exists. Performing split...")
        metadata = perform_split(args)
        assert os.path.exists(args['metadata_file'])
    return metadata
    
        
def get_data_loaders(args, metadata):
    """
    Gets the Greenstand data loaders, ready for input to the network
    """
    print("Creating data loaders...")
    # Load up the dataset and split into train, val, test
    trainset, valset, testset = load_datasets(args, metadata)
    if args['visualize'] == 'y':
        visualize_images(trainset)

    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=args['num_workers'])
    valloader = torch.utils.data.DataLoader(valset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=args['num_workers'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'],
                                              shuffle=True, num_workers=args['num_workers'])
    
    # Get Dataset Attributes
    dataset_attributes = {
        'n_train': len(trainset),
        'n_val': len(valset), 
        'n_test': len(testset),
        'n_classes': len(trainset.classes),  
        'lr_schedule': [40, 50, 60], 
        'class2num_instances': {
            'train': trainset.get_num_instances_per_class(),
            'val': valset.get_num_instances_per_class(),
            'test': testset.get_num_instances_per_class()
        },
        'class_to_idx': trainset.class_to_idx,
        'invalid_classes' : metadata['invalid'],
        'total_instances_per_class': metadata['total_instances_per_class']
    }
    
    return trainloader, valloader, testloader, dataset_attributes  


def load_preloaded_model(args, dataset_attributes):
    """
    Given a location for a pre-trained model, loads it with PyTorch
    """
    model = plantnet_utils.get_model(args, n_classes=1081)
    
    g_args = vars(args)
    if g_args['preloaded_model_location'] !="None" and g_args['preloaded_model_location'] != "":
        model.load_state_dict(torch.load(g_args['preloaded_model_location'])['model'])
            
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        
         # Replace final layer with new fc layer
        if g_args['model'] == 'inception_v4':
            model.aux_logits = False
            model.last_linear = torch.nn.Linear(in_features=1536, out_features=dataset_attributes['n_classes'], bias=True)
        else:
            model.fc = torch.nn.Linear(2048, dataset_attributes['n_classes'])
        
        model.cuda()
    
    return model


def load_preloaded_model_prediction(args, dataset_attributes):
    """
    Given a location for a pre-trained model, loads it with PyTorch
    """
    g_args = vars(args)
    model = plantnet_utils.get_model(args, n_classes=dataset_attributes['n_classes'])
    model.load_state_dict(torch.load(g_args['preloaded_model_location'])['model'])
    return model


# Classes
class GreenstandLabelledImageDataset(Dataset):
    """
    Contains functions that have the ability to load the Dataset into memory and input into the NN
    """
    def __init__(self, base_dirs, transform, images=None):
        """
        base_dirs -> if there are multiple local paths to all the images, provide them here
            ie: ['path/to/haiti', 'path/to/herbarium']
        """
        # Load up pre-sorted images from metadata
        if images is not None:
            self.images = images
            labels = list(set(images[:, 1]))
                
        else:
            self.images = []
            labels = []
            
            for directory in base_dirs:  # haiti
                for label in os.listdir(directory):  # list(haiti) -> [ACACAURI...]
                    fname = f"{directory}/{label}"
                    if os.path.isdir(fname):  # ACACAURI -> True
                        for file in os.listdir(fname):  # list(haiti/ACAURI/) -> [image.jpg...]
                            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jfif'):
                                self.images.append((fname + "/" + file, label))
                                labels.append(label)
            self.images = np.array(self.images)
        
        self.classes = sorted(list(set(labels)))
        
        self.class_to_idx = {}
        for i in range(len(self.classes)):
            self.class_to_idx[self.classes[i]] = i
        
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.images[idx][0])  # np.asarray(Image.open(self.images[idx][0]))
        label = self.images[idx][1]

        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[label]
            
        return image, label
    
    def get_num_instances_per_class(self):
        num_instances = {}
        labels = [x[1] for x in self.images]
        for c in self.classes:
            num_instances[self.class_to_idx[c]] = labels.count(c)
        return num_instances

    
class MaxCenterCrop:
    def __call__(self, sample):
        min_size = min(sample.size[0], sample.size[1])
        return CenterCrop(min_size)(sample)

def visualize_imbalance(dataset_attributes):
    """
    Show number of instances in each class (show excluded classes too)
    """
    # Get the number of classes - each is a tick mark
    coords = np.array(list(range(dataset_attributes['n_classes'] + len(dataset_attributes['invalid_classes']))))
    
    # Get the number of instances in each class
    width = 1.00  # the width of the bars
    heights_train = list(dataset_attributes['class2num_instances']['train'].values()) + [0 for i in range(len(dataset_attributes['invalid_classes']))]
    heights_val = list(dataset_attributes['class2num_instances']['val'].values()) + [0 for i in range(len(dataset_attributes['invalid_classes']))]
    heights_test = list(dataset_attributes['class2num_instances']['test'].values()) + [0 for i in range(len(dataset_attributes['invalid_classes']))]
    
    # Color classes that were modeled green, others red
    colors = ['g' for i in range(dataset_attributes['n_classes'])] + ['r' for i in range(len(dataset_attributes['invalid_classes']))]
    labels = list(dataset_attributes['class_to_idx'].keys()) + dataset_attributes['invalid_classes']
    
    # Generate plot
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    fig.set_dpi(150)
    fig.set_facecolor('white')
    
    # Create bars for each data split
    rects_train = ax.bar(coords - width/4, heights_train, width/4, label='Train')
    rects_val = ax.bar(coords, heights_val, width/4, label='Val')
    rects_test = ax.bar(coords + width/4, heights_test, width/4, label='Test')
    
    ax.bar_label(rects_train, padding=3)
    ax.bar_label(rects_val, padding=3)
    ax.bar_label(rects_test, padding=3)

    # Set labels/legends
    ax.set_ylabel('Num Images')
    ax.set_title('Images per Species')
    # plt.bar_label(ax.containers[0])
    ax.set_xticks(coords)
    ax.set_xticklabels(labels, rotation = 45, ha='right')
    plt.legend()
    plt.show()
    

def visualize_images(dataset):
    # Collect images
    images = []
    for c in dataset.classes:
        image_file = dataset.images[np.where(dataset.images[:, 1]==c)][0][0]
        images.append((image_file,c))
   # Set standards
    width = 20
    height = 20
    fig = plt.figure(figsize=(8, 8))
    fig.set_facecolor('white')
    columns = 4
    rows = round(len(images) / columns)
    # Create images
    for i in range(len(images)):
        image_file = images[i][0]
        spec = images[i][1]
        img = Image.open(image_file)
        ax = fig.add_subplot(rows, columns, i+1)
        ax.title.set_text(spec)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
    plt.show()


def grid_search(args, param_space):
    """
    Does a grid search of all items included in param space
    param_space should be a dict whose key is the hyperparameter name (ie: lr) and values is a list of values to search through (ie: [0.1, 0.01, 0.05])
    """
    all_results = {}
#     for key in param_space
    
    
#     for adam_opt in ['y','n']:
#         for focal_opt in ['y','n']:
#             for lr in [.001, .005, .01, .05, .1]:
#                 for mu in [0.0, .01]:
#                     i+=1
#                     if i < skip_to or mu > 0.0:
#                         continue
#                     print("---------------------------------------------- NEW ----------------------------------------------------")
#                     config = load_config_file(hyperparameter_config_file='hyperparameters.yaml')
#                     config['use_adam_optimizer'] = adam_opt
#                     config['use_focal_loss'] = focal_opt
#                     config['lr'] = lr
#                     config['mu'] = mu

#                     arg_list = get_args(config)
#                     parser = argparse.ArgumentParser()
#                     cli.add_all_parsers(parser)
#                     args = parser.parse_args(args=arg_list)
#                     acc = train(args)

#                     print(f"RESULT: ADAM:{adam_opt} FOCAL:{focal_opt} LR:{lr} MU:{mu} - ACC:{acc}")
#                     all_results[(adam_opt, focal_opt, lr, mu)] = acc
#     print(all_results)