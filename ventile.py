import boto3
import os
import sys
from PIL import Image
import io
import json
import shutil

CLIENT_ACCESS_KEY="AKIAQYWVSSHAHHFC6ZWX"
CLIENT_SECRET_KEY = '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'
project_dir = "/home/olivier/greenstand/freetown/"
assert (os.path.exists(project_dir))
project_good_dir =  project_dir + "good/"
project_bad_dir = project_dir + "bad/"
project_rejected_dir = project_dir + "rejected/"
PROBA_THRESHOLD = 0.9
COPY = True

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

    endpoint_name = "haiti4"

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

## Loops over subfolders of freetown to fetch all images
## Seaparates those with a trustable probability from the others, and see if they are well classified
def ventile():
    total = 0.0
    true_pos = 0.0
    false_pos = 0.0
    discarded = 0.0
    # Loops over freetown folder
    for name in os.listdir(project_dir):
      f = os.path.join(project_dir, name)
      if not os.path.isdir(f):
        continue
      species_groundtruth = os.path.basename(f) # The expected species
      # Loops over the images of this species
      for filename in os.listdir(f):
        if not os.path.isfile( os.path.join(f, filename)):
          continue
        # Calls endpoint
        the_file = os.path.join(f, filename)
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
        if (max_prob > PROBA_THRESHOLD):
          if species_groundtruth == max_species:
            true_pos += 1
            if COPY:
               shutil.copy(the_file, project_good_dir + species_groundtruth + "_" + str(true_pos) + ".jpg")
          else:
            false_pos += 1
            if COPY:
               shutil.copy(the_file, project_bad_dir + species_groundtruth + "_" + max_species +  "_" + str(false_pos) + ".jpg")

        else:
          discarded += 1
          if COPY:
            shutil.copy(the_file, project_rejected_dir + species_groundtruth + "_" + str(discarded) + ".jpg")
        remaining = total - discarded
        if (total%10 == 0 and remaining > 0):
          print("accuracy: ", true_pos/remaining, ", % of rejected: ", discarded/total)
          # TODO copy to folder according to result

def main():
    try:
        ventile()
    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())