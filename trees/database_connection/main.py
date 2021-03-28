#!/usr/bin/python

import os
import argparse
import db_connection
import cvat_task_manager
from config import config

"""
This script requires to have a separate file called "credentials.ini" with credentials to create the ssh tunnel, 
to connect to the database and to create cvat tasks.

[postgresql]
host=localhost
database=<database>
user=<user>
port=<local_port>
password=<password>

[sshtunnel]
ssh_address_or_host=<tunnel_host>
ssh_username=<user for tunnel connection>
ssh_password=<remote_gateway_password>
ssh_port=<tunnel_port>
ssh_pkey=<path_to_private_key>
remote_bind_address=<remote_server>
remote_bind_address_port=<remote_port>
local_bind_address=localhost
local_bind_address_port=<local_port>

[cvat]
auth=<cvat_user>
password=<cvat_password>
host=<cvat_host>
port=<cvat_port>
"""


def get_haiti_trees_query(last_img_time):

    return "SELECT uuid, time_created, date_part('epoch', time_created), image_url \
            FROM trees  \
            WHERE planter_id IN (select id from planter where organization_id = 194) AND date_part('epoch', time_created) > " \
            + str(last_img_time) \
            + "ORDER BY date_part('epoch', time_created) ASC;"

"""
def get_haiti_trees_species_query():

    return "SELECT DISTINCT tree_species.name \
           FROM trees \
	       JOIN tree_species ON species_id = tree_species.id \
           WHERE planter_id IN (select id from planter where organization_id = 194);"
"""

def update_query_log(file_name,results):
    new_last_timestamp = int(results[-1][2])

    with open(file_name, "a") as f:
        f.write(str(new_last_timestamp) + "\n")


def create_new_cvat_task_list(results):

    img_urls = []

    for tree in results:
        img_urls.append(tree[3])

    return img_urls


cvat_cli_script = "/home/dalsa90/projects/cvat/utils/cli/cli.py"


def get_haiti_species():

    default_labels = ["HAS_TREE", "NO_TREE"]
    haiti_species = ["OTHER", "PERSAMER", "CATALONG", "CEDRODOR", "MANGINDI"]

    return default_labels + haiti_species

    """
    return [ \
        {"name": "OTHER", "attributes": []}, \
        {"name": "PERSAMER", "attributes": []}, \
        {"name": "CATALONG", "attributes": []}, \
        {"name": "CEDRODOR", "attributes": []}, \
        {"name": "MANGINDI", "attributes": []} ]
    """


def create_json_labels(working_folder, species):

    import json
    json_labels_file_name = os.path.join(working_folder, "haiti_labels.json")

    species_list = []

    for iPlant in species:
        species_list.append({"name": iPlant, "attributes": []})

    with open(json_labels_file_name, "w") as f:
        json.dump(species_list, f, indent=4)

    return json_labels_file_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    working_folder = '/home/dalsa90/projects/greenstand/greenstand_data_analysis/trees/database_connection/'

    parser.add_argument('--restart-logging', type=bool, default=False, help='boolean to delete previous logging and start pulling image urls from scratch')
    parser.add_argument('--test-images', type=str, default="all", help='Number of images to test with, "all" for all images')
    parser.add_argument('--call-cvat-cli', type=bool, default=False, help='"all for including every class"')
    parser.add_argument('--data', type=str, default=None, help='"all for including every class"')
    parsed_args = parser.parse_args()

    log_file_name = working_folder + "log_queries.txt"
    img_urls_file_name = working_folder + "img_urls.txt"

    test_images = parsed_args.test_images

    if test_images != "all":
        test_images = int(test_images)

    if parsed_args.restart_logging:

        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        if os.path.exists(img_urls_file_name):
            os.remove(img_urls_file_name)
        if os.path.exists(img_urls_file_name):
            os.remove(os.path.join(working_folder, "haiti_labels.json"))

    if os.path.exists(log_file_name):
        with open(log_file_name, "r") as f:
            this_timestamp = 0

            # read all logged timestamps
            lines = f.readlines()
            # extract last timestamp used to
            # create a task and remove '\n'
            last_timestamp = lines[-1][:-1]

    else:
        last_timestamp = "0"
        with open(log_file_name, "a") as f:
            f.write(last_timestamp + "\n")

    credentials_file = working_folder + 'credentials.ini'
    tunnel = db_connection.create_ssh_tunnel(credentials_file)
    tunnel.start()
    (connection, cur) = db_connection.connect(credentials_file)

    haiti_trees_query = get_haiti_trees_query(last_timestamp)
    results = db_connection.execute_query(cur, haiti_trees_query)

    if len(results) > 0:

        images_urls = create_new_cvat_task_list(results)
        update_query_log(log_file_name, results)

        with open(img_urls_file_name, "a") as f:
            if test_images != "all":
                images_urls = images_urls[:test_images]

            for this_url in images_urls:
                f.write(this_url + "\n")

    db_connection.close_connection(connection, cur)
    tunnel.close()

    if parsed_args.call_cvat_cli:

        haiti_species = get_haiti_species()
        json_labels_file_name = create_json_labels(working_folder, haiti_species)

        cvat_params = config(credentials_file, 'cvat_local')
#        cvat_task_manager.get_current_cvat_tasks(cvat_params)
        cvat_task_manager.create_new_cvat_task(cvat_params, images_urls, json_labels_file_name)


