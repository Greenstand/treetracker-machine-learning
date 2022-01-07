#!/usr/bin/python

import os
import sys
import argparse
import db_connection
from config import config
from datetime import date

from manage_log_file import LogFileManager

cvat_base_dir = os.path.abspath('./cvat_greenstand')  # should point to the installation of cvat
sys.path.insert(1, os.path.join(cvat_base_dir, 'cvat/utils/cli'))

import cvat_task_manager

"""
This script requires to have a separate file called "database.ini" with credentials to create the ssh tunnel, 
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

def create_new_cvat_task_list(results):

    img_urls = []

    for tree in results:
        img_urls.append(tree[3])

    return img_urls


default_labels = ["HAS_TREE", "NO_TREE", "SOIL", "UNKNOWN"]

def get_haiti_species():

    haiti_species = ["ACACAURI", "ANACOCCI", "ANNOMURI", "ANNOSQUA", \
                     "ARTOALTA", "CATALONG", "CEDRODOR", "CITR0000", \
                     "COCONUCI", "COLUARBO", "CORDALLI", "INDE0002", \
                     "INDE0003", "INGAFEUI",  "MANGINDI", "MANIZAPO", \
                     "MORIOLEI", "PERSAMER", "PSIDGUAJ", "SIMAGLAU", \
                     "TAMAINDI", "TERMCATA", "THEOCACA"]

    return default_labels + haiti_species


def create_json_labels(working_folder, species):

    import json
    json_labels_file_name = os.path.join(working_folder, "haiti_labels.json")

    species_list = []

    for iPlant in species:
        species_list.append({"name": iPlant, "attributes": []})

    with open(json_labels_file_name, "w") as f:
        json.dump(species_list, f, indent=4)

    return (json_labels_file_name, species_list)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--restart-logging', type=int, default=0, help='boolean to delete previous logging and start pulling image urls from scratch')
    parser.add_argument('--test-images', type=str, default="all", help='Number of images to test with, "all" for all images')
    parser.add_argument('--cvat-cli', type=str, default='./cli.py', help='Path to cli.py cvat script')
    parser.add_argument('--images-per-cvat-task', type=int, default=0, help='int: number of images per cvat task.')
    parser.add_argument('--working-folder', type=str, default='./', help='path to folder where databases.ini is and for saving new files.')
    parser.add_argument('--prefix', type=str, default='', help='prefix to be appended.')
    parser.add_argument('--assign-to', type=str, default='', help='Assignee for the task.')
    parsed_args = parser.parse_args()

    working_folder = parsed_args.working_folder
    restart_logging = parsed_args.restart_logging

    img_urls_file_name = os.path.join(working_folder, parsed_args.prefix + "img_urls.txt")
    haiti_labels_file_name = os.path.join(working_folder, "haiti_labels.json")

    images_per_task = parsed_args.images_per_cvat_task

    if images_per_task <= 0:
        raise Exception("Images per task must be positive.")

    assignee = parsed_args.assign_to

    if restart_logging:
        if os.path.exists(img_urls_file_name):
            os.remove(img_urls_file_name)
        if os.path.exists(img_urls_file_name):
            os.remove(haiti_labels_file_name)

    log_file_name = "log_queries.txt"

    log_file_manager = LogFileManager(working_folder, restart_logging, log_file_name)
    last_timestamp = log_file_manager.get_last_timestamp()

    credentials_file = os.path.join(working_folder, 'database.ini')
    tunnel = db_connection.create_ssh_tunnel(credentials_file)
    tunnel.start()
    (connection, cur) = db_connection.connect(credentials_file)

    haiti_trees_query = get_haiti_trees_query(last_timestamp)
    results = db_connection.execute_query(cur, haiti_trees_query)

    db_connection.close_connection(connection, cur)
    tunnel.close()

    if len(results) > 0:

        test_images = parsed_args.test_images

        if test_images != "all":
            test_images = int(test_images)
        else:
            test_images = len(results)

        # Remove images which do not fit new task.
        # They'll be used for next task creation.
        test_images = test_images - (test_images % images_per_task)
        results = results[:test_images]

        numOfNewTasks = int(test_images / images_per_task)

        images_urls = create_new_cvat_task_list(results)
        log_file_manager.update_query_log(results)

        with open(img_urls_file_name, "a") as f:
            for this_url in images_urls:
                f.write(this_url + "\n")

        cvat_params = config(credentials_file, 'cvat_local')

        cvat_mgr = cvat_task_manager.CvatManager(parsed_args.cvat_cli, cvat_params)
        cvat_mgr.get_current_cvat_tasks()

        if len(images_urls) > 0:
            haiti_species = get_haiti_species()
            (labels_json_file, labels) = create_json_labels(working_folder, haiti_species)
            task_prefix = parsed_args.prefix + "haiti_" + str(date.today()) + "_" + "_task_"

            for i in range(numOfNewTasks):
                task_name = task_prefix + str(i)
                cvat_mgr.create_new_cvat_task(images_urls[i*images_per_task:(i+1)*images_per_task ], labels_json_file, task_name)

    else:
        print("No images left in the db.")
