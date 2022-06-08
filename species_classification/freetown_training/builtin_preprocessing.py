'''
This script is intended to be an example of using transforms in Sagemaker. 

See how this works at:
https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html

Created by Shubhom
June 2022
'''
# PyTorch Libraries
from torchvision import transforms
import argparse
import pathlib
import os
import pandas as pd
import shutil
from xml.etree import ElementTree
from PIL import Image
import numpy as np
import logging


def get_train_test_inds(y, train_proportion=0.7, seed=42):
    '''
    Generates indices, making random stratified split into training set and testing sets with proportions train_proportion and (1-train_proportion) of 
    initial sample. y is any iterable indicating classes of each observation in the sample. Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    
    @param y (Iterable): Iterable of columns ordered by dataset index (i.e. dataset[i, label] = y[i])
    @param train_proportion (float): Portion of data to keep as training data 
    @param seed (int): Random seed 
    '''
    
    np.random.seed(seed)
    y = np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))
        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True
    return train_inds,test_inds

def create_path_lists(input_path, val_split, test_split, seed=42):
    '''
    Create train/val/test split based on paths. Make val_split 0 if using later cross-validation method. 
    Uses get_train_test_inds to stratify sampling by class.
    
    @param input_path(str): dataset location on Sagemaker machine local
    @param val_split (float): Portion of total dataset to use as left-out validation  (between 0 and 1)
    @param test_split(float): Portion of total dataset to use as test set (between 0 and 1)
    @param seed(int): Random seed for reproducing splits in the future
    '''
    imgs = {}
    for class_name in CLASSES:
        if not os.path.exists(os.path.join(input_path, class_name)):
            logging.info("Skipping class %s"%class_name, " couldn't find it")
            continue
        temp_imgs = os.listdir(os.path.join(input_path, class_name))
        logging.info("create_path_lists found %s images in class %d"%(class_name, len(temp_imgs)))
        for img_path in temp_imgs:
            if not "tar" in img_path:
                name = os.path.splitext(img_path)[0]
                full_path = os.path.join(input_path, class_name, img_path)
                imgs[name] = (class_name, full_path) 
                # later, we can modify this to change how non-tree species are classified. 
                    
    imgs = pd.DataFrame.from_dict(imgs, orient="index")
    imgs.columns = ["class", "full_path"]
    logging.info ("Img paths preview: ")
    logging.info (str(imgs.head(5)))
    total_size = imgs.shape[0]    
    logging.info ("Total num images: " + str(total_size))
    nontest_idxs, test_idxs = get_train_test_inds(imgs.iloc[:, 0], train_proportion=1-val_split, seed=seed)
    train_idxs, val_idxs = get_train_test_inds(imgs.iloc[nontest_idxs, 0], train_proportion=1-(1/(1-test_split) * val_split), seed=seed)
    
    return imgs.iloc[nontest_idxs, :][train_idxs], imgs.iloc[nontest_idxs, :][val_idxs], imgs[test_idxs]


    
        
def image_transforms(transform_params):
    '''
    returns torchvision pre-processing pipeline
    TODO: define some image transform 
    '''
    preprocessing = transforms.Compose([
        transforms.Resize((transform_params["resize"], transform_params["resize"])),
    ])
    
    return preprocessing



def image_augmentation(img, transform_params):
    '''
    TODO: define some image augmentations based on class imbalances
    '''
    img = img.resize((transform_params["resize"], transform_params["resize"]))  
    img += np.random.normal(0, 1, (img.size[0], img.size[1], 3)) # num channels should be 3
    return Image.fromarray(np.uint8(img))
    
def save_from_dataframe(df, output_dir, transform_params):
    '''
    Take a DataFrames produced by create_path_lists above and generates directories. Sagemaker transfers the contents of this local directory upon job 
    completion to S3. 
    
    @param df (pd.DataFrame): DataFrame containing columns ["class", "full_path"]
    @param output_dir (str): Path to save output to 
    '''
    saved_images = {}
    preprocessing_fn = image_transforms(transform_params)
    for class_name in CLASSES:
        df_subset = df[df["class"] == class_name]
        class_output_path = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
        for row in df_subset.itertuples():
            name = row.Index
            img = Image.open(row.full_path)
            img = preprocessing_fn(img)
            img.save(os.path.join(class_output_path, name + ".jpg"))
            saved_images[name] = os.path.join(class_output_path, name + ".jpg")
    saved_images = pd.DataFrame.from_dict(saved_images, orient="index")
    saved_images.columns = ["path"]
    return saved_images
    
    
        

def augment_from_dataframe(df, output_dir, transform_params, suffix="_ aug", fraction_augmented=0.4):
    '''
    Perform augmentation similar to save_from_dataframe but with a suffix for augmented images and a predefined subsampling of images to augment, if 
    desirable. 
    
    @param df (pd.DataFrame): DataFrame containing columns ["class", "full_path"]
    @param output_dir (str): Path to save output to 
    @param suffix (str): A suffix to identify augmented images in the output directory
    @param num (int): Number of images to augment per class
    '''
    # decide on augmentation rule (balance classes, preserve class distro)
    augmented_images = {}
    for class_name in CLASSES:
        df_class = df[df["class"] == class_name]
        class_output_path = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_path):
            raise ValueError("This class hasn't been created yet un-augmented.")
        if isinstance(fraction_augmented, float):
            frac = fraction_augmented
        elif isinstnace(fraction_augmented, list):
            assert len(fraction_augmented) == len(CLASSES)
            frac = fraction_augmented[CLASSES.indexof(class_name)]
        for j in range(int(frac * df_class.shape[0])):
            random_idx = np.random.randint(low=0, high=df_class.shape[0]) # sample an image at random
            row = df_class.iloc[random_idx, :] # weird indexing on purpose to get DF output, not Series
            name = row.index[0]
            if j==0: 
                print (name)
            img = Image.open(row.full_path)
            img = image_augmentation(img, transform_params)
            if os.path.exists(os.path.join(class_output_path, name + suffix + ".jpg")):
                suffix += "_x"
            img.save(os.path.join(class_output_path, name + suffix + ".jpg"))
            augmented_images[name] = os.path.join(class_output_path, name + suffix + ".jpg")
    augmented_images = pd.DataFrame.from_dict(augmented_images, orient="index")
    augmented_images.columns df.columns
    return augmented_images

def preprocess(args):
    '''
    A  main method  
    '''    
    OUTPUT_TRAIN_PATH = os.path.join(PROCESSING_DIR, args.output_path_train)
    OUTPUT_VALIDATION_PATH = os.path.join(PROCESSING_DIR, args.output_path_validation)
    OUTPUT_TEST_PATH = os.path.join(PROCESSING_DIR, args.output_path_test)
    
    if 1 - args.val_split_ratio - args.test_split_ratio <= 0 or args.val_split_ratio > 1 or args.test_split_ratio > 1:
        raise ValueError("Poor splits defined. Check tr/val/test split hyperparams")
    
    training_paths, validation_paths, test_paths = create_path_lists(INPUT_PATH, 
                                                                     val_split=args.val_split_ratio, 
                                                                     test_split=args.test_split_ratio)
    
    
    transform_params = {"resize": args.resize}
    save_from_dataframe(training_paths, OUTPUT_TRAIN_PATH, transform_params)
    augment_from_dataframe(training_paths, OUTPUT_TRAIN_PATH, transform_params)
    save_from_dataframe(validation_paths, OUTPUT_VALIDATION_PATH, transform_params)
    save_from_dataframe(test_paths, OUTPUT_TEST_PATH, transform_params)

def listdirs(rootdir):
    ''' 
    Helper to list subdirectories of a directory
    '''
    subdirs = []
    for it in os.scandir(rootdir):
        if it.is_dir():
            subdirs.append(it.name)
    return subdirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-split-ratio', type=float, default=0.2)
    parser.add_argument('--test-split-ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument ('--resize', type=int, default=224)
    parser.add_argument('--input_path', type=str, default="raw")
    parser.add_argument('--output_path_train', type=str, default="train")
    parser.add_argument('--output_path_validation', type=str, default="validation")
    parser.add_argument('--output_path_test', type=str, default="test")
    PROCESSING_DIR = "/opt/ml/processing/"
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    INPUT_PATH = os.path.join(PROCESSING_DIR, args.input_path)
    CLASSES = listdirs(INPUT_PATH) 
    for c in CLASSES:
        d = os.path.join(INPUT_PATH, c)
        logging.info (str(c) + ": " + str(len(os.listdir(d))))
    logging.info("Classes: " + str(CLASSES))
    preprocess(args)