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

import numpy as np
import os.path as osp
import math

class Imdb(object):
    """
    Base class for dataset loading
    Parameters:
    ----------
    name : str
        name of dataset
    """
    def __init__(self, name):
        self.name = name
        self.classes = []
        self.num_classes = 0
        self.image_set_index = None
        self.num_images = 0
        self.labels = None
        self.padding = 0

    def image_path_from_index(self, index):
        """
        load image full path given specified index
        Parameters:
        ----------
        index : int
            index of image requested in dataset
        Returns:
        ----------
        full path of specified image
        """
        raise NotImplementedError

    def label_from_index(self, index):
        """
        load ground-truth of image given specified index
        Parameters:
        ----------
        index : int
            index of image requested in dataset
        Returns:
        ----------
        object ground-truths, in format
        numpy.array([id, xmin, ymin, xmax, ymax]...)
        """
        raise NotImplementedError

    def get_annotated_imglist(self, root=None):
        """
        save imglist to disk
        Parameters:
        ----------
        fname : str
            saved filename
        """
        def progress_bar(count, total, suffix=''):
            import sys
            bar_len = 24
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100.0 * count / float(total), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
            sys.stdout.flush()

        str_list = []

        index = 0

        for cls in range(self.num_subfolders):

            str_list.append([])

            for path in self.images_path_per_class[cls]:
                progress_bar(index, self.num_images)
                label = self.label_from_index(index)

                if label.size < 1:
                    continue

                path = osp.join(self.images_path, path+self.extension)

                if root:
                    path = osp.relpath(path, root)

                str_list[-1].append('\t'.join([str(2), str(label.shape[1])] \
                  + ["{0:.4f}".format(x) for x in label.ravel()] + [path,]) + '\n')

                index += 1

        return str_list

    def split_imglist(self, splitting, root=None):
        """
        splitting must be a list of ratios to split the full dataset in training validation and testing,
        they should sum to 1
        returns a list, where each element is a list of observations.
        """

        train = splitting[0]
        val   = splitting[1]
        test  = splitting[2]

        assert train+val+test == 1

        annotated_image_list = self.get_annotated_imglist(root)

        train_str_list = []
        val_str_list   = []
        test_str_list  = []

        train_idx = 0
        val_idx   = 0
        test_idx  = 0

        for cls in annotated_image_list:

            num_training_img   = math.floor(train * len(cls))
            num_validation_img = math.floor(val * len(cls))

            for idx in range(num_training_img):
                train_str_list.append(str(train_idx) + '\t' + cls[idx])
                train_idx += 1

            for idx in range(num_training_img, num_training_img+num_validation_img):
                val_str_list.append(str(val_idx) + '\t' + cls[idx])
                val_idx += 1

            for idx in range(num_training_img + num_validation_img, len(cls)):
                test_str_list.append(str(test_idx) + '\t' + cls[idx])
                test_idx += 1

        str_list = [train_str_list, val_str_list, test_str_list]

        return str_list


    def save_imglist(self, str_list, fname=None, shuffle=False):
        """
        save imglist to disk
        Parameters:
        ----------
        fname : str
            saved filename
        """

        if str_list:
            if shuffle:
                import random
                random.shuffle(str_list)
            if not fname:
                fname = self.name + '.lst'
            with open(fname, 'w') as f:
                for line in str_list:
                    f.write(line)
        else:
            raise RuntimeError("No images in imdb")

    def _load_class_names(self, filename, dirname):
        """
        load class names from text file
        Parameters:
        ----------
        filename: str
            file stores class names
        dirname: str
            file directory
        """
        full_path = osp.join(dirname, filename)
        classes = []
        with open(full_path, 'r') as f:
            classes = [l.strip() for l in f.readlines()]
        return classes


