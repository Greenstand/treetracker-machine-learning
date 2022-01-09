
import argparse
import os
import sys

annotation_tools_base_dir = os.path.abspath('./greenstand/greenstand_data_analysis/annotation_tools/')
sys.path.append(os.path.join(annotation_tools_base_dir))
from annotation_tools_core.config import config

cvat_base_dir = os.path.abspath('./cvat_greenstand')  # should point to the installation of cvat
sys.path.append(os.path.join(cvat_base_dir, 'cvat/utils/cli'))

import cvat_task_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cvat-cli', type=str, default='./cli.py', help='Path to cli.py cvat script')
    parser.add_argument('--working-folder', type=str, default='./',
                        help='path to folder where databases.ini is and for saving new files.')
    parser.add_argument('--target-folder', type=str, default='./',
                        help='path to folder where to dump annotations.')

    parsed_args = parser.parse_args()

    working_folder = parsed_args.working_folder
    target_folder = parsed_args.target_folder

    credentials_file = os.path.join(working_folder, 'database.ini')
    cvat_params = config(credentials_file, 'cvat_local')

    project_dir = os.path.abspath('.')

    cvat_mgr = cvat_task_manager.CvatManager(parsed_args.cvat_cli, cvat_params)
    completed_tasks = cvat_mgr.get_current_cvat_tasks(status='completed')

    for done_task in completed_tasks:
        cvat_mgr.dump_target_annotations(done_task['id'], os.path.join(target_folder, "output_" + str(done_task['id']) + ".zip"))
