import boto3
import os
import sys
from PIL import Image
import io
import json
import shutil
import random

# Please create a config file and fill it with
# CLIENT_ACCESS_KEY = 'your client access key'
# CLIENT_SECRET_KEY = 'your client secret key'
from config import *

# Folder where all test images can be found
project_dir = "/home/olivier/greenstand/data/"
assert (os.path.exists(project_dir))
project_good_dir =  "/home/olivier/greenstand/good/" # Where the well classified images will be copied
project_bad_dir =  "/home/olivier/greenstand/bad/" # Where the wrongly classified images will be copied
project_rejected_dir =  "/home/olivier/greenstand/rejected/" # Where the ambiguous images will be copied
PROBA_THRESHOLD = 0.9 # If max(probas) < PROBA_THRESHOLD, then we'll consider the image as ambiguous
COPY = True # Whether we want to copy images in project_good/bad/rejected_dir, for visual inspection
num_cases = 400 # Number of images from project_dir to classify, if we don't want to test everything

## Inference runtime
runtime = boto3.client("sagemaker-runtime", 
    region_name="us-east-1", # added this line in local version of this code
    aws_access_key_id=CLIENT_ACCESS_KEY,
    aws_secret_access_key=CLIENT_SECRET_KEY,
                      )

## Infers probabilities given an image path. Returns probas as json (empty if something goes wrong)
def infer(name_img):
  my_json = {}
  try:
    # change this to point to your image
    path_to_img = os.path.join(project_dir, name_img)
    img = Image.open(path_to_img)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    endpoint_name = "freetown1" #"haiti4"

    content_type = "image/jpeg"
    payload = img_byte_arr

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Body=payload
    )
    my_json = json.loads(response['Body'].read())
  except:
    pass
  return my_json

# Return pairs of image path and the ground truth species
def prepare_data():
    paths = []
    species = []
    for name in os.listdir(project_dir):
      f = os.path.join(project_dir, name)
      if not os.path.isdir(f):
        continue
      species_groundtruth = os.path.basename(f) # The expected species
      # Loops over the images of this species
      for filename in os.listdir(f):
        if not os.path.isfile( os.path.join(f, filename)):
          continue
        the_file = os.path.join(f, filename)
        paths.append(the_file)
        species.append(species_groundtruth)
    return paths, species

## Loops over subfolders of freetown to fetch all images
## Seaparates those with a trustable probability from the others, and see if they are well classified
def ventile():
    total = 0.0
    true_pos = 0.0
    false_pos = 0.0
    discarded = 0.0

    # Gather all images and labels
    paths, species = prepare_data()

    # Shuffle data
    num_files = len(paths)
    idxs = [i for i in range(num_files)]
    random.shuffle(idxs)
    num_files = min(num_files, num_cases) # Limits the number of images to test

    # Loops over freetown folder
    for i in range(num_files):
      idx = idxs[i]
      the_file = paths[idx]
      species_groundtruth = species[idx]
      #print(str(i) + ' ' + str(idx) +  ' ' + species_groundtruth)

      # Calls endpoint
      result = infer(the_file)
      total += 1

      # Looks for max proba in the result
      max_prob = 0.0
      max_species = ''
      for k,v in result.items():
        if (float(v) > max_prob):
          max_prob = float(v)
          max_species = k

      # Checks if the highest proba is over our chosen threshold
      if (max_prob > PROBA_THRESHOLD): # Result is non ambiguous, let's see now if it is right
        if species_groundtruth == max_species:
          true_pos += 1
          if COPY:
              shutil.copy(the_file, project_good_dir + species_groundtruth + "_" + str(true_pos) + ".jpg")
        else:
          false_pos += 1
          if COPY:
               shutil.copy(the_file, project_bad_dir + species_groundtruth + "_" + max_species +  "_" + str(false_pos) + ".jpg")

      else: # Result is ambiguous, and we consider that it is because the quality of the image is not sufficient
        discarded += 1
        if COPY:
          shutil.copy(the_file, project_rejected_dir + species_groundtruth + "_" + str(discarded) + ".jpg")
      remaining = total - discarded
      if (total%10 == 0 and remaining > 0):
        print("accuracy: ", true_pos/remaining, ", % of rejected: ", 100.*discarded/total)

def main():
    try:
        ventile()
    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())