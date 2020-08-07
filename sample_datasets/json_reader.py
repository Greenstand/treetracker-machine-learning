import json
import argparse


def read_check(jsonfile):
    with open(jsonfile) as f:
        bs = json.load(f)
    print (bs['title'])
    print (bs['coordinates'])
    print (bs['quality'])
    print (bs['duplicates'])
    print (bs['pattern'])
    print ("If these fields look alright, file is OK")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="path to .json file")
    args = parser.parse_args()
    read_check(args.json_path)