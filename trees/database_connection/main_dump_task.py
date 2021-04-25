
import cvat_task_manager
import argparse
from config import config
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cvat-cli', type=str, default='./cli.py', help='Path to cli.py cvat script')
    parser.add_argument('--working-folder', type=str, default='./',
                        help='path to folder where databases.ini is and for saving new files.')

    parsed_args = parser.parse_args()

    working_folder = parsed_args.working_folder

    credentials_file = working_folder + 'database.ini'
    cvat_params = config(credentials_file, 'cvat_local')

    project_dir = os.path.abspath('../../../../')
    target_folder = os.path.join(project_dir, "greenstand/sample_images/dumped_annotations")

    cvat_mgr = cvat_task_manager.CvatManager(os.path.join(project_dir, "cvat/utils/cli/cli.py"), cvat_params)
    completed_tasks = cvat_mgr.get_current_cvat_tasks(status='completed')

    cvat_mgr.dump_target_annotations(completed_tasks[0]['id'], os.path.join(target_folder, "output_" + str(completed_tasks[0]['id']) + ".zip"))
