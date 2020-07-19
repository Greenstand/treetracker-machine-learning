# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:12:53 2020

@author: A-Kees.Brekelmans
"""

import json
from disk_planter import word_to_dir

with open("word_net_ids.json") as handle:
    word_net_dic = json.load(handle)

print(f"Categories: {word_net_dic.keys()}")    

    
word_to_dir(word_net_dic) 
