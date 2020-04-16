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




