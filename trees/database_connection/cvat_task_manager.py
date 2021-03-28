
import os

cvat_cli_script = "/home/dalsa90/projects/cvat/utils/cli/cli.py"

def create_new_cvat_task(cvat_params, img_urls, json_labels_file_name, task_name="my_newtask"):

    all_image_urls = ""

    for img in img_urls:
        all_image_urls += img + " "

    cvat_call = cvat_cli_script \
                + ' --auth ' + cvat_params['auth'] + ':' + cvat_params['password'] + ' '\
                + '--server-host ' + cvat_params['host'] + ' '\
                + '--server-port ' + cvat_params['port'] + ' '\
                + 'create ' + task_name + ' ' \
                + '--labels ' + json_labels_file_name + ' ' \
                + 'remote ' + all_image_urls

    os.system(cvat_call)


def get_current_cvat_tasks(cvat_params):

    cvat_call = cvat_cli_script \
                + ' --auth ' + cvat_params['auth'] + ':' + cvat_params['password'] + ' '\
                + '--server-host ' + cvat_params['host'] + ' '\
                + '--server-port ' + cvat_params['port'] + ' '\
                + 'ls'

    os.system(cvat_call)
