
import numpy as np
import pandas as pd
import os
import urllib
import time

import requests
import json
import datetime
import logging




def pipe_transfer(df_row):
    syscall = "wget  \"%s\" -q | aws s3 cp %s %s --quiet"%(df_row["url"], df_row["imname"], os.path.join(s3_dest, df_row["class"], df_row["imname"]))
    code = os.system(syscall)
    if code == 0:
        delete_call = "find . -name \"*.jpg\" -type f -delete"
        os.system(delete_call) # remove wget artifact
    return code


# Directly downloads the dataurl variable (link to Cam's training set) to datadir
datadir = "/home/ec2-user/data/freetown_training.psv"
dataurl =  "https://raw.githubusercontent.com/Greenstand/Tree_Species/master/training/training_freetown_tagged.psv"
data_update = requests.get(dataurl)

if data_update.status_code == 200:
    print ("Successfully downloaded training set")
    with open(datadir, 'w') as f:
        f.write(data_update.text)

baseurl = "https://treetracker-production-images.s3.eu-central-1.amazonaws.com/"

assert os.path.exists(datadir)
data = pd.read_csv(datadir, sep="|", header=None)
print (data.head(5))
data.columns = ["class", "imname"]
data["url"] = baseurl + data["imname"]
print (data.shape[0], " samples")

original_data_bucket = "treetracker-training-images"
dataset_key = "freetown" # use this to restrict to a particular directory

s3_dest = 's3://{}/{}/'.format(original_data_bucket, dataset_key)

rows_done = 66494

print ("Starting full dataset S3 transfer")
start = datetime.datetime.now()
data = data.iloc[rows_done:, :]
data["s3_transfer_successful"] = data.apply(pipe_transfer, axis=1)
print ("Finished in " , datetime.datetime.now() - start)
print (data[data["s3_transfer_successful"]], " samples downloaded out of ", data.shape[0])

