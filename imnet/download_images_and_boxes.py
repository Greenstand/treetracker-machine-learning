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

if __name__ == "__main__":
    datadir = os.path.join(os.path.dirname(os.getcwd()), "data", "imnet", "original_images")
    bbdir = os.path.join(os.path.dirname(os.getcwd()), "data", "imnet", "bounding_boxes")
    synsets = {
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


    print("These downloads are big and may take some time... please be patient :)")
    for title, wnid in synsets.items():
        save_dir = os.path.join(datadir, title)
        t = time.time()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
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
            print ("Seems like %s raw images have  already been downloaded. If you wish to redownload, delete the original directory with the corresponding wnid %s"%(title, wnid))
        bb_url = "http://www.image-net.org/downloads/bbox/bbox/%s.tar.gz"%wnid
        bb_savedir = os.path.join(bbdir, title)
        if not os.path.exists(bb_savedir):
            os.mkdir(bb_savedir)
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
        else:
            print("Seems like the %s  annotations have already been downloaded. If you wish to redownload, delete the original directory with the corresponding wnid. " % wnid)
        time_elapsed = time.time() - t
        print("Raw image data and bounding boxes for %s (wnid %s)  finished in  %s seconds" % (
                title, wnid, time_elapsed))
