import requests
import os
import urllib.request

class DataLoader():
    def __init__(self, dir, server_url):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.download_dir = dir
        self.url = server_url
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64)  Chrome/72.0.3626.119 Safari/537.36"}

    def put_hash(self, img, hash):
        pass

    def get_bounding_images(self, long1, long2, lat1, lat2):
        # TODO: Syntax in requests documentation passes args via "params" arg in requests.get; a little different with lists so clarify this
        return requests.get(self.url + "?bounds=%f,%f,%f,%f"%(long1, lat1, long2, lat2), headers=self.headers)

    def download_image(self, url, fname):
        with open(fname, "w+") as f:
            f.write(urllib.request.urlretrieve(url, fname))


if __name__ == "__main__":
    name = "bengal"
    data_dir = os.path.join(os.path.dirname(os.getcwd()),"data", name)
    loader = DataLoader(dir=data_dir, server_url="http://167.172.211.46:3007/captures/")
    output =  loader.get_bounding_images(37.47144102360772, -3.261007479439057, 37.4703209331231, -3.2624149647584013).json()['data']
    print (output[0].keys())
    for o in output:
        fname = os.path.join(data_dir,os.path.splitext(o["image_url"][1]))
        metadata = {"id": o["id"],
             "planter_id": o["planter_id"],
             "image_url": o["image_url"],
             "lat": o["lat"],
             "lon": o["lon"],
             "est_geometric_loc": o["estimated_geometric_location"],
            "fname": fname
             }
    with open(fname, "w+") as f:
        f.write(urllib.request.urlretrieve(url, fname))


        # print (o['image_url'])
    # "lat":"22.975806","lon":"88.453996","gps_accuracy":34,