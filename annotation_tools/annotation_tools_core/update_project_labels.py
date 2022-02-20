import json

import xml.etree.ElementTree as ET
from datetime import datetime
import os
import urllib.request

default_labels = ["HAS_TREE",
                  "NO_TREE",
                  "SOIL",
                  "UNKNOWN"
                  ]

def download_raw_tree_species(tree_project_folder):
    tree_species_url = "https://raw.githubusercontent.com/Greenstand/Tree_Species/master/tree_species.xml"
    print("Downloading herbarium.")

    raw_tree_species_file = os.path.join(tree_project_folder, "raw_haiti_tree_species.xml")

    herbarium_filename, headers = urllib.request.urlretrieve(tree_species_url, filename=raw_tree_species_file)
    print("Herbarium downloaded.")

    return herbarium_filename

def update_project_labels(working_folder, project_name):
    """Create a json label file with the updated tree species from the herbarium."""

    tree_project_folder = os.path.join(working_folder, "tree_projects/")

    if not os.path.exists(tree_project_folder):
        os.makedirs(tree_project_folder)

    raw_tree_species_filename = os.path.abspath(download_raw_tree_species(tree_project_folder))

    print("Abs path to raw species is " + raw_tree_species_filename)

    # Pass the path of the xml document
    tree = ET.parse(raw_tree_species_filename)

    # get the parent tag
    root = tree.getroot()

    # List all label
    herbarium_species = default_labels

    for child in root:
        if child.tag == 'mtype':
            herbarium_species.append(child.attrib['id'])

    species_to_be_dumped = []

    # For each new species, create a dictionary to be dumped in the json file
    for species in herbarium_species:
        species_to_be_dumped.append( {"name": species, "attributes": []} )

    now = datetime.now()
    target_file_name = project_name + "_labels_" + now.strftime("%Y%m%d_%H%M%S") + ".json"
    target_file_path = os.path.join(tree_project_folder, target_file_name)

    species_json_string = json.dumps(species_to_be_dumped, indent=4)
    with open(target_file_path, 'w') as outfile:
        outfile.write(species_json_string)

    return target_file_path