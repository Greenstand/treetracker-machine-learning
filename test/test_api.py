import unittest
import numpy as np
import time
import imagehash
from PIL import Image
import os
from api.dataloader import DataLoader



class CorrectAPI(unittest.TestCase):
    def test_kilema(self):
        name = "kilema_tanzania"
        data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", name)
        loader = DataLoader(dir=data_dir, name=name, server_url="http://167.172.211.46:3007/captures/")
        loader.retrieve_dataset(37.47144102360772, -3.261007479439057, 37.4713209331231, -3.2624149647584013,
                                create_md=True)
    def test_rajasthan(self):
        name = "rajasthan"
        data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", name)
        loader = DataLoader(dir=data_dir, name=name, server_url="http://167.172.211.46:3007/captures/")
        loader.retrieve_dataset(75.54368166588793,27.107339191431038,75.53993129411211,27.10598456572285,
                                create_md=True)
    def test_lotan(self):
        name = "lotan_israel"
        data_dir = os.path.join(os.path.dirname(os.getcwd()), "data", name)
        loader = DataLoader(dir=data_dir, name=name, server_url="http://167.172.211.46:3007/captures/")
        loader.retrieve_dataset(35.08995660177582,29.989065255077037,35.08245585822417,29.986429132771992,
                                create_md=True)



