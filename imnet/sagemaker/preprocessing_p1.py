'''
This script is intended to be an example of using transforms in Sagemaker. 

See how this works at:
https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html

Created by Shubhom
December 2020
'''
# PyTorch Libraries
import argparse
import pathlib
import os
import pandas as pd
import shutil
from xml.etree import ElementTree
from PIL import Image
import numpy as np

def write_out(dataloader, output_dir):
    pass

def parse_annotation(filepath):
    if not os.path.exists(filepath):
        return None
    else:
        if os.path.splitext(filepath)[1] == ".xml":
            with open(filepath) as file_obj:
                tree = ElementTree.parse(file_obj)
                root = tree.getroot()
                obj = root.find("object")
                b = obj.find("bndbox")
                xmin = int(b.find("xmin").text)
                ymin = int(b.find("ymin").text)
                xmax = int(b.find("xmax").text)
                ymax = int(b.find("ymax").text)
                return xmin, ymin, xmax, ymax
        else:
            return None

        import numpy as np  

def get_train_test_inds(y, train_proportion=0.7, seed=42):
    '''
    Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
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
    classes = SYNSETS.keys()
    imgs = {}
    for class_name in classes:
        if not os.path.exists(os.path.join(input_path, "original_images", class_name)):
            print ("Skipping class %s"%class_name)
            continue
        temp_imgs = os.listdir(os.path.join(input_path, "original_images", class_name))
        for img_path in temp_imgs:
            if not "tar" in img_path:
                name = os.path.basename(img_path.split('.')[0])
                annotation_path = os.path.join(input_path, "bounding_boxes", class_name, "Annotation", name.split("_")[0], f"{name}.xml")
                box = parse_annotation(annotation_path)
                if box is None:
                    annotation_path = None
                imgs[name] = (class_name, box, img_path, annotation_path) # later, we can modify this to change how non-tree species are classified. 
                    
    imgs = pd.DataFrame.from_dict(imgs, orient="index")
    imgs.columns = ["class", "bbox", "full_path", "annotation_path"]
    print ("Img paths preview: ")
    print (imgs.head(5))
    total_size = imgs.shape[0]
    train_idxs, test_idxs = get_train_test_inds(imgs.loc[:, ["class"]], train_proportion=1-test_split, seed=seed)
    train_idxs, val_idxs = get_train_test_inds(imgs.loc[train_idxs, ["class"]], train_proportion=1-(1/(1-test_split) * val_split), seed=seed)
    return imgs.loc[train_idxs, :], imgs.loc[val_idxs, :], imgs.loc[test_idxs, :]


    
        
def image_transform(img):
    '''
    TODO: define some image transform 
    '''
    img = Image.resize(img, (64, 64)) 
    return img
    

def image_augmentation(img):
    '''
    TODO: define some image augmentations based on class imbalances
    '''
    img = Image.resize(img, (64, 64))  
    
    return img
    
def save_from_dataframe(df, output_dir):
    '''
    Take the DataFrames produced above and generate directories
    
    '''
    for class_name in SYNSETS.keys():
        df = df[df["class"] == class_name]
        class_output_path = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
        for i, row in df.iterrows():
            name = row.index()
            img = Image.open(row["full_path"])
            img = image_transform(img)
            Image.save(img, class_output_path + name + ".JPEG")
    
        

def augment_from_dataframe(df, output_dir, suffix="_ aug"):
    # decide on augmentation rule (balance classes, preserve class distro)
    for class_name in SYNSETS.keys():
        df = df[df["class"] == class_name]
        class_output_path = os.path.join(output_dir, class_name)
        if not os.path.exists(class_output_path):
            raise ValueError("This class hasn't been created yet un-augmented.")
        for i, row in df.iterrows():
            name = row.index()
            img = Image.open(row["full_path"])
            img = image_augmentation(img)
            Image.save(img, class_output_path + name + suffix + ".JPEG")
            
    
            
def preprocess(args):
    INPUT_PATH = os.path.join(PROCESSING_DIR, args.input_path)
    
    OUTPUT_TRAIN_PATH = os.path.join(PROCESSING_DIR, args.output_path_train)
    OUTPUT_VALIDATION_PATH = os.path.join(PROCESSING_DIR, args.output_path_validation)
    OUTPUT_TEST_PATH = os.path.join(PROCESSING_DIR, args.output_path_test)
    
    if 1 - args.val_split_ratio - args.test_split_ratio <= 0 or args.val_split_ratio > 1 or args.test_split_ratio > 1:
        raise ValueError("Poor splits defined. Check tr/val/test split hyperparams")
    
    training_paths, validation_paths, test_paths = create_path_lists(INPUT_PATH, 
                                                                     val_split=args.val_split_ratio, 
                                                                     test_split=args.test_split_ratio)
    
    
    save_from_dataframe(training_paths, OUTPUT_TRAIN_PATH)
    augment_from_dataframe(training_paths, OUTPUT_TRAIN_PATH)
    save_from_dataframe(validation_paths, OUTPUT_VALIDATION_PATH)
    save_from_dataframe(test_paths, OUTPUT_TEST_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-split-ratio', type=float, default=0.2)
    parser.add_argument('--test-split-ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--input_path', type=str, default="raw")
    parser.add_argument('--output_path_train', type=str, default="train")
    parser.add_argument('--output_path_validation', type=str, default="validation")
    parser.add_argument('--output_path_test', type=str, default="test")
    PROCESSING_DIR = "/opt/ml/processing/"
    args = parser.parse_args()
    
        # Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
    tree_synsets = {
        "judas": "n12513613",
        "palm": "n12582231",
        "pine": "n11608250",
        "china tree": "n12741792",
        "fig": "n12401684",
        "cabbage": "n12478768",
        "cacao": "n12201580",
        "kapok": "n12190410",
        "iron": "n12317296",
        "linden": "n12202936",
        "pepper": "n12765115",
        "rain": "n11759853",
        "dita": "n11770256",
        "alder": "n12284262",
        "silk": "n11759404",
        "coral": "n12527738",
        "huisache": "n11757851",
        "fringe": "n12302071",
        "dogwood": "n12946849",
        "cork": "n12713866",
        "ginkgo": "n11664418",
        "golden shower": "n12492106",
        "balata": "n12774299",
        "baobab": "n12189987",
        "sorrel": "n12242409",
        "Japanese pagoda": "n12570394",
        "Kentucky coffee": "n12496427",
        "Logwood": "n12496949"
    }
    nontree_synsets = {
        "garbage_bin": "n02747177",
        "carion_fungus": "n13040303",
        "basidiomycetous_fungus": "n13049953",
        "jelly_fungus": "n13060190",
        "desktop_computer": "n03180011",
        "laptop_computer": "n03642806",
        "cellphone": "n02992529",
        "desk": "n03179701",
        "station_wagon": "n02814533",
        "pickup_truck": "n03930630",
        "trailer_truck": "n04467665"
    }
    SYNSETS = {**tree_synsets, **nontree_synsets}
    preprocess(args)