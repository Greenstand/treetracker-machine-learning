import argparse
import os
import sys

annotation_tools_base_dir = os.path.abspath('./greenstand/greenstand_data_analysis/annotation_tools/')
sys.path.append(os.path.join(annotation_tools_base_dir))
from annotation_tools_core.config import config

cvat_base_dir = os.path.abspath('./cvat_greenstand')  # should point to the installation of cvat
sys.path.insert(1, os.path.join(cvat_base_dir, 'cvat/utils/cli'))
import cvat_task_manager

def get_tak_ids_from_tasks(tasks):

    task_ids = []
    for task in tasks:
        task_ids.append(task['id'])

    return task_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cvat-cli', type=str, default='./cli.py', help='Path to cli.py cvat script')
    parser.add_argument('--working-folder', type=str, default='./',
                        help='path to folder where databases.ini is and for saving new files.')
    parser.add_argument('--delete', type=str, default='none',
                        help='Which tasks to be deleted, options are: "all", "completed" or a task id number')
    parser.add_argument('--confirm-deletion', type=str, default='completed',
                        help='Tasks to delete: "completed", for deleting all completed tasks, or a task id number')

    parsed_args = parser.parse_args()

    working_folder = parsed_args.working_folder
    tasks_to_be_deleted = parsed_args.delete
    confirm_deletion = parsed_args.confirm_deletion

    credentials_file = os.path.join(working_folder, 'database.ini')
    cvat_params = config(credentials_file, 'cvat_local')

    cvat_mgr = cvat_task_manager.CvatManager(parsed_args.cvat_cli, cvat_params)

    if tasks_to_be_deleted == "completed":
        tasks_to_be_deleted = cvat_mgr.get_current_cvat_tasks(status='completed')
        task_ids_to_be_delete = get_tak_ids_from_tasks(tasks_to_be_deleted)
    else:
        task_ids_to_be_delete = [tasks_to_be_deleted]

    for task_id in task_ids_to_be_delete:

        yes_no = 'y'

        if confirm_deletion:
            print("Are you sure you want to delete task_id = " + str(task_id) + "? y/n")
            yes_no = str(input())
        if yes_no == 'y':
            cvat_mgr.delete_cvat_task(task_id)



