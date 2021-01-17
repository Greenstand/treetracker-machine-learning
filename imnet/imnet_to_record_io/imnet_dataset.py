# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
import xml.etree.ElementTree as ET
import cv2


class ImnetDb(Imdb):
    """
    Implementation of Imdb for Pascal VOC datasets
    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    year : str
        year of dataset, can be 2007, 2010, 2012...
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """

    def __init__(self, dataset_path, tree_vs_nontree, shuffle=False, is_train=False,
                 names='imnet.names'):

        # init_name = 'voc_' + year + '_' + image_set
        # init_name = 'imnet_' + image_set
        init_name = 'imnet_dataset'
        super(ImnetDb, self).__init__(init_name)
        # self.image_set = image_set
        self.year = "2020"
        self.devkit_path = dataset_path
        self.data_path = dataset_path
        self.extension = '.JPEG'
        self.bb_extension = '.xml'
        self.is_train = is_train

        self.images_path = 'original_images'
        self.bb_path = 'bounding_boxes'

        self.subfolders = self._load_class_names(names, os.path.dirname(__file__))

        if tree_vs_nontree:
            self.classes = ['tree', 'nontree']
        else:
            # if each subfolder defines a class
            self.classes = self.subfolders

        self.config = {'use_difficult': True,
                       'comp_id': 'comp4', }

        self.num_classes = len(self.classes)
        self.num_subfolders = len(self.subfolders)

        # text file with all the relative path to the images
        images_relative_path_filename = self.get_files_recursively(dataset_path, self.subfolders, self.images_path,
                                                                   self.bb_path,
                                                                   self.extension,
                                                                   self.bb_extension)

        self.image_set_index = self._load_image_set_index(shuffle, images_relative_path_filename)

        self.num_images = len(self.image_set_index)

        if self.is_train:
            self.labels = self._load_image_labels()

    @property
    def cache_path(self):
        """
        make a directory to store all caches
        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path

    def _load_image_set_index(self, shuffle, data_filenames_path):
        """
        find out which indexes correspond to given image set (train or val)
        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """

        # image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Main', self.image_set + '.txt')

        image_set_index_file = os.path.join(self.data_path, data_filenames_path)

        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)

        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]

        if shuffle:
            np.random.shuffle(image_set_index)

        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, self.images_path, name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth
        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path
        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, self.bb_path, index + '.xml')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths
        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)

            label = get_labels_from_file(label_file, self.classes)

            temp.append(label)
        return temp

    def get_files_recursively(self, dataset_folder, classnames, images_path, bb_path, extension, bb_extension):

        # given FOLDER, write a file with all file names contained in FOLDER, recursively

        from os import listdir
        from os.path import isfile, join

        images_abs_path = os.path.join(dataset_folder, images_path)
        bb_abs_path = os.path.join(dataset_folder, bb_path)

        output_filename = join(dataset_folder, 'images.txt')

        self.images_path_per_class = []

        with open(output_filename, 'w') as output_file:
            for subfolder in classnames:

                onlyfiles = []

                for f in listdir(join(bb_abs_path, subfolder)):

                    if isfile(join(bb_abs_path, subfolder, f)) and (bb_extension in f):

                        # remove the extension for convenience
                        f = f.replace(bb_extension, '')

                        if corresponding_image_exists(f, images_abs_path, subfolder, extension):
                            onlyfiles.append(join(subfolder, f))
                    else:
                        continue

                self.images_path_per_class.append(onlyfiles)

                for item in onlyfiles:
                    output_file.write("%s\n" % item)

        print("Finished generating output file", )

        return output_filename

    def _get_imsize(self, im_name):
        """
        get image size info
        Returns:
        ----------
        tuple of (height, width)
        """
        img = cv2.imread(im_name)
        return (img.shape[0], img.shape[1])


def get_labels_from_file(label_file, classes):

    tree = ET.parse(label_file)
    root = tree.getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    label = []

    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text)
        # if not self.config['use_difficult'] and difficult == 1:
        #     continue
        cls_name_idx = obj.find('name').text
        cls_name = convert_to_tree_nontree(cls_name_idx)

        if cls_name not in classes:
            continue

        cls_id = classes.index(cls_name)
        xml_box = obj.find('bndbox')
        xmin = float(xml_box.find('xmin').text) / width
        ymin = float(xml_box.find('ymin').text) / height
        xmax = float(xml_box.find('xmax').text) / width
        ymax = float(xml_box.find('ymax').text) / height

        if xmax <= xmin or ymax <= ymin:
            print("Found an example with xmax <= xmin or ymax <= ymin ", obj)
        else:
            label.append([cls_id, xmin, ymin, xmax, ymax])

    return np.array(label)

def convert_to_species(cls_name):
    if cls_name == 'n12513613':
        return 'judas'
    if cls_name == 'n12582231':
        return 'palm'
    if cls_name == 'n11608250':
        return 'pine'
    if cls_name == 'n12741792':
        return 'china tree'
    if cls_name == 'n12401684':
        return 'fig'
    if cls_name == 'n12478768':
        return 'cabbage'
    if cls_name == 'n12201580':
        return 'cacao'
    if cls_name == 'n12190410':
        return 'kapok'
    if cls_name == 'n12317296':
        return 'iron'
    if cls_name == 'n12202936':
        return 'linden'
    if cls_name == 'n12765115':
        return 'pepper'
    if cls_name == 'n11759853':
        return 'rain'
    if cls_name == 'n11770256':
        return 'dita'
    if cls_name == 'n12284262':
        return 'alder'
    if cls_name == 'n11759404':
        return 'silk'
    if cls_name == 'n12527738':
        return 'coral'
    if cls_name == 'n11757851':
        return 'huisache'
    if cls_name == 'n12302071':
        return 'fringe'
    if cls_name == 'n12946849':
        return 'dogwood'
    if cls_name == 'n12713866':
        return 'cork'
    if cls_name == 'n11664418':
        return 'ginkgo'
    if cls_name == 'n12492106':
        return 'golden shower'
    if cls_name == 'n12774299':
        return 'balata'
    if cls_name == 'n12189987':
        return 'baobab'
    if cls_name == 'n12242409':
        return 'sorrel'
    if cls_name == 'n12570394':
        return 'Japanese pagoda'
    if cls_name == 'n12496427':
        return 'Kentucky coffee'
    if cls_name == 'n12496949':
        return 'Logwood'
    else:
        return cls_name

def convert_to_tree_nontree(cls_name):
    if cls_name == 'n12513613':
        return 'nontree' # 'judas'
    if cls_name == 'n12582231':
        return 'tree' # 'palm'
    if cls_name == 'n11608250':
        return 'tree' # 'pine'
    if cls_name == 'n12741792':
        return 'tree' # 'china tree'
    if cls_name == 'n12401684':
        return 'nontree' # 'fig'
    if cls_name == 'n12478768':
        return 'tree' # 'cabbage'
    if cls_name == 'n12201580':
        return 'nontree' # 'cacao'
    if cls_name == 'n12190410':
        return 'tree' # 'kapok'
    if cls_name == 'n12317296':
        return 'tree' # 'iron'
    if cls_name == 'n12202936':
        return 'tree' # 'linden'
    if cls_name == 'n12765115':
        return 'tree' # 'pepper'
    if cls_name == 'n11759853':
        return 'nontree' # 'rain'
    if cls_name == 'n11770256':
        return 'tree' # 'dita'
    if cls_name == 'n12284262':
        return 'nontree' #'alder'
    if cls_name == 'n11759404':
        return 'nontree' # 'silk'
    if cls_name == 'n12527738':
        return 'nontree' #'coral'
    if cls_name == 'n11757851':
        return 'huisache'
    if cls_name == 'n12302071':
        return 'tree' # 'fringe'
    if cls_name == 'n12946849':
        return 'nontree' # 'dogwood'
    if cls_name == 'n12713866':
        return 'nontree' # 'cork'
    if cls_name == 'n11664418':
        return 'tree' # 'ginkgo'
    if cls_name == 'n12492106':
        return 'tree' # 'golden shower'
    if cls_name == 'n12774299':
        return 'tree' # 'balata'
    if cls_name == 'n12189987':
        return 'tree' # 'baobab'
    if cls_name == 'n12242409':
        return 'tree' # 'sorrel'
    if cls_name == 'n12570394':
        return 'tree' # 'Japanese pagoda'
    if cls_name == 'n12496427':
        return 'tree' # 'Kentucky coffee'
    if cls_name == 'n12496949':
        return 'tree' # 'Logwood'
    else:
        return cls_name

def corresponding_image_exists(f, images_folder, subfolder, images_extension):
    return os.path.exists(os.path.join(images_folder, subfolder, f) + images_extension)