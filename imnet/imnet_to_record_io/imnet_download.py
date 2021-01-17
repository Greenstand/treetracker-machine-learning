import argparse
import requests
import os
import time
import tarfile

'''
Important! The credentials here are private, please do not share these any external party
'''
_USER = "shubhomb"
_KEY = "a057c7178f6ba153a61e3e2ec546ea3b32a8d463"

def download_imnet_dataset(target_folder, synsets):

    if target_folder is None:
        target_folder = os.path.join(os.path.dirname(os.getcwd()), "data", "imnet" )

    # datadir specifies where raw images should be downloaded
    datadir = os.path.join(target_folder, "original_images")

    #bbdir specifies where annotation XMLs should be downloaded
    bbdir = os.path.join(target_folder, "bounding_boxes")

    if synsets is None:
        from imnet_classes import get_default_synsets
        synsets = get_default_synsets()

    print("These downloads are big and may take some time... please be patient :)")

    for title, wnid in synsets.items():
        save_dir = os.path.join(datadir, title)
        t = time.time()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            savefile = os.path.join(save_dir, wnid + ".tar.gz")
            url = "http://www.image-net.org/download/synset?wnid=%s&username=%s&accesskey=%s&release=latest&src=stanford" % (
            wnid, _USER, _KEY)
            item = requests.get(url)

            if item.status_code == 200:
                print("Raw image: status code 200: success")
            else:
                print("Raw image: status code ", item.status_code)
            with open(savefile, 'wb') as f:
                f.write(item.content)
            tarf = tarfile.open(savefile)
            tarf.extractall(save_dir)
        else:
            print ("Seems like %s raw images have already been downloaded. If you wish to redownload, delete the original directory with the corresponding wnid %s"%(title, wnid))

        bb_url = "http://www.image-net.org/downloads/bbox/bbox/%s.tar.gz"%wnid
        bb_savedir = os.path.join(bbdir, title)

        if not os.path.exists(bb_savedir):
            os.makedirs(bb_savedir)
            item = requests.get(bb_url)
            if item.status_code == 200:
                print("Bounding box: status code 200: success")
            else:
                print("Bounding box: status code ", item.status_code)
            bb_annotations = os.path.join(bb_savedir, wnid + ".tar.gz")
            with open(bb_annotations, 'wb') as f:
                f.write(item.content)
            tarf = tarfile.open(bb_annotations)
            tarf.extractall(bb_savedir)

            # place xml in title/ subfolder
            annotations_dir = os.path.join(bb_savedir, "Annotation", wnid)

            for filename in os.listdir(annotations_dir):
                os.rename(os.path.join(annotations_dir,filename), os.path.join( os.path.join(bb_savedir,filename ) ))
            import shutil
            shutil.rmtree(os.path.join(bb_savedir, "Annotation"))

        else:
            print("Seems like the %s  annotations have already been downloaded. If you wish to redownload, delete the original directory with the corresponding wnid. " % wnid)
        time_elapsed = time.time() - t
        print("Raw image data and bounding boxes for %s (wnid %s)  finished in  %s seconds" % (
                title, wnid, time_elapsed))

import argparse

if __name__ == "__main__":

    # This main downloads the full dataset.

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--folder', type=str, default=None,
                        help='Target folder for storing dataset')

    args = parser.parse_args()

    synsets = None
    download_imnet_dataset(args.folder, synsets)