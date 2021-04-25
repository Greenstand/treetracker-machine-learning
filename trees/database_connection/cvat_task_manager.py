import os
import sys

base_dir = os.path.abspath('../../../../')
sys.path.insert(1, os.path.join(base_dir, 'cvat/utils/'))

import logging
import requests
import sys
from http.client import HTTPConnection
from cli.core.core import CLI, CVAT_API_V1
log = logging.getLogger(__name__)


def config_log(level):
    log = logging.getLogger('core')
    log.addHandler(logging.StreamHandler(sys.stdout))
    log.setLevel(level)
    if level <= logging.DEBUG:
        HTTPConnection.debuglevel = 1


class CvatManager:

    def __init__(self, cvat_cli_script, cvat_params):
        self.cvat_cli_script = cvat_cli_script
        self.cvat_params = cvat_params
        self.cvat_params['https'] = False

    def get_calling_args(self):
        args = {'auth': (self.cvat_params['auth'], self.cvat_params['password']),
                'server_host': self.cvat_params['server_host'],
                'server_port': self.cvat_params['server_port'],
                'https': self.cvat_params['https'],
                'loglevel': 20,
                'use_json_output': False}

        return args


    def get_action(self,target_action):

        actions = {'create': CLI.tasks_create,
                   'delete': CLI.tasks_delete,
                   'ls': CLI.tasks_list,
                   'frames': CLI.tasks_frame,
                   'dump': CLI.tasks_dump,
                   'upload': CLI.tasks_upload}

        return actions[target_action]


    def create_new_cvat_task(self, img_urls, json_labels_file_name, task_name):
        all_image_urls = ""

        for img in img_urls:
            all_image_urls += img + " "

        cvat_call = self.cvat_cli_script \
                    + ' --auth ' + self.cvat_params['auth'] + ':' + self.cvat_params['password'] + ' ' \
                    + '--server-host ' + self.cvat_params['server_host'] + ' ' \
                    + '--server-port ' + self.cvat_params['server_port'] + ' ' \
                    + 'create ' + task_name + ' ' \
                    + '--labels ' + json_labels_file_name + ' ' \
                    + 'remote ' + all_image_urls

        os.system(cvat_call)


    def get_current_cvat_tasks(self, status=[]):
        # Possible values for flag are:
        # empty to return all tasks
        # 'annotation' for tasks which have not been entirely annotated
        # 'validation' for tasks which have been annotated but not validated
        # 'completed' for tasks which have been completed (annotated + validated)

        args = self.get_calling_args()
        all_tasks = self.call_cvat_api('ls', args)

        if status == []:
            return all_tasks

        output_tasks = []

        for task in all_tasks:
            if task['status'] == status:
                output_tasks.append(task)

        return output_tasks


    def dump_target_annotations(self, target_task, filename):

        args = self.get_calling_args()
        args['task_id'] = target_task
        args['filename'] = filename
        args['fileformat'] = 'CVAT for images 1.1'

        self.call_cvat_api('dump', args)

        return None


    def call_cvat_api(self, target_action, args):

        target_call = self.get_action(target_action)
        config_log(args['loglevel'])

        with requests.Session() as session:
            api = CVAT_API_V1('%s:%s' % (args['server_host'], args['server_port']), args['https'])
            cli = CLI(session, api, args['auth'])
            try:
                #actions['ls'](cli, **args.__dict__)
                output = target_call(cli, **args)
            except (requests.exceptions.HTTPError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException) as e:
                log.critical(e)

        return output

