import argparse
import api.dataloader as DL

''' 
CLI-interface for data download. Usage like: 

python terminal.py "C:/Users/shubh/OneDrive/Documents/greenstand/data/lotan_israel/" "lotan_israel" -coordinates 35.0880657601578 29.988515416223 3
5.08431538838197 29.98719735652034
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="directory to download images to",
                        type=str)
    parser.add_argument("name", help="dataset name",
                        type=str)

    parser.add_argument('-coordinates', '--coords', nargs=4, help='<Required> Lat-Long bounding box', type=float)
    args = parser.parse_args()
    loader = DL.DataLoader(dir=args.data_dir, name=args.name, server_url="http://167.172.211.46:3007/captures/")
    loader.retrieve_dataset(args.coords[0], args.coords[1], args.coords[2], args.coords[3],
                            create_md=True)
    print ("Complete!")
