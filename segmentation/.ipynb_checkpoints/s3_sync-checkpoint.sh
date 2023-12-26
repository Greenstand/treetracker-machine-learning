#!/bin/bash
aws s3 sync s3://treetracker-training-images/pilot_annotations/PlantVillage/ local_data/
python3 split_dataset.py 