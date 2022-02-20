import argparse
import os, sys

annotation_tools_base_dir = os.path.abspath('./greenstand/greenstand_data_analysis/annotation_tools/')
sys.path.append(os.path.join(annotation_tools_base_dir))
from annotation_tools_core.update_project_labels import update_project_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--working-folder', type=str, default='./',
                        help='path to folder for downloding html file of species and saving the new file used ' +
                             'for defining the labels in the task.')

    parser.add_argument('--project-name', required=True, type=str,
                        help='Project name. The output file is called "project_name_labels_timestamp.json".' )

    parsed_args = parser.parse_args()
    working_folder = parsed_args.working_folder
    project_name = parsed_args.project_name
    update_project_labels(working_folder, project_name)
