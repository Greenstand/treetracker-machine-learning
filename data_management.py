import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import lmdb
import pickle
import os

"""
Exploratory file for image loading and duplicate detection
"""


class GreenstandDataset():
    """
    A way to get images and quickly refer to them in the file system
    """
    def __init__(self, dir):
        self.datadir = dir

    def populate(self):
        self.data = pd.read_csv(self.datadir, engine="python")

    def render_image(self, url):
        '''
        For the url, transform the tree image to a numpy ndarra
        :param url: (str) url of image provided in csv
        :return: (ndarray)
        '''
        response = requests.get(url)
        return np.array(Image.open(BytesIO(response.content)))

    def save_one_img(self, lmdb_dir, image, id):
        '''
        Stores a single image to a LMDB.
            Parameters:
            ---------------
            image       image array, (32, 32, 3) to be stored
            image_id    integer unique ID for image
            label       image label
        '''
        map_size = image.nbytes * 10

        # Create a new LMDB environment
        env = lmdb.open(str(lmdb_dir), map_size=map_size)

        # Start a new write transaction
        with env.begin(write=True) as txn:
            # All key-value pairs need to be strings
            txn.put(id.encode("ascii"), pickle.dumps(image))
        env.close()

    def write_database(self,lmdb_dir, max_entries, begin_idx=0, ids=None):
        """
        Based off https://realpython.com/storing-images-in-python/
        :param lmdb_dir:
        :param max_entries:
        :return:
        """
        map_size = 1e8
        # Create a new LMDB environment
        env = lmdb.open(str(lmdb_dir), map_size=map_size)
        if ids is None:
            with env.begin(write=True) as txn:
                    for j in range(max_entries): # could itreate through all entries theoretically but need full memory
                        # Start a new write transaction
                        # All key-value pairs need to be strings
                        txn.put(str(self.data.loc[j + begin_idx]["id"]).encode("ascii"),pickle.dumps(self.render_image(self.data.loc[j]["image_url"])))
            env.close()
        else: # write a database of the given ids
            with env.begin(write=True) as txn:
                for id_num in ids:
                    samp = self.data[self.data["id"] == id_num]
                    if samp is None:
                        print ("ID %d not found " %id_num)
                    else:
                        print (samp["image_url"].values[0])
                        txn.put(str(id_num).encode("ascii"), pickle.dumps(self.render_image(samp["image_url"].values[0])))
            env.close()
        print ("Database written to %s"%lmdb_dir)


    def read_image_from_db(self,lmdb_dir,key):
        env = lmdb.open(str(lmdb_dir),readonly=True)
        with env.begin() as txn:
            data = pickle.loads(txn.get(str(key).encode("ascii")))
        return data

if __name__ == "__main__":
    data = GreenstandDataset('nov11data.csv')
    data.populate() # necessary if writing to database
    lmdb_path = os.path.join(os.getcwd(),"duplicates_db")
    data.write_database(lmdb_path,max_entries=25, ids=list(range(93457, 93482)))

    a = data.read_image_from_db(lmdb_path, data.data.iloc[0][0])
    b = data.read_image_from_db(lmdb_path, 93457)


