
import requests
import os
import json
import urllib.request

class DataLoader():
    def __init__(self, name, dir, server_url):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.name = name
        self.download_dir = dir
        self.url = server_url
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)  Chrome/72.0.3626.119 Safari/537.36"}


    def get_bounding_images(self, long1, long2, lat1, lat2):
        return requests.get(self.url, params={"bounds": [long1, lat1, long2, lat2]}, headers=self.headers)

    def download_image(self, url, id):
        fpath = os.path.join(self.download_dir, str(id)) # file corresponding to this id
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        return urllib.request.urlretrieve(url, os.path.join(fpath, str(id) +  os.path.splitext(url)[1]))



    def retrieve_dataset(self, long1, long2, lat1, lat2, create_md=False):
        output = self.get_bounding_images(long1, long2, lat1, lat2).json()['data']
        for o in output:
            metadata = {"id": o["id"],
                        "planter_id": o["planter_id"],
                        "image_url": o["image_url"],
                        "lat": o["lat"],
                        "lon": o["lon"],
                        "est_geometric_loc": o["estimated_geometric_location"],
                        "hash": o["matching_hash"]
                        }
            self.download_image(o["image_url"], o["id"])
            if create_md:
                with open(os.path.join(self.download_dir, str(o["id"]), (str(o["id"]) + ".txt")), "w+") as f:
                    json.dump(metadata,f)

    def put_hash(self, id, hash=0):
        return requests.put(self.url + id + "/hash/", json={"hash": hash})

