from imnet_download import download_imnet_dataset
from imnet_dataset import ImnetDb
import subprocess
import sys, os
import argparse


curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

def select_synsets(target_synsets):

    # target_synsets must be a list with the species we want to download and operate with

    from imnet_classes import get_default_synsets
    default_synsets = get_default_synsets()

    if target_synsets == 'all':
        return default_synsets
    else:
        assert isinstance(target_synsets, (list, str))

        synsets = dict()

        for item in target_synsets:
            synsets[item] = default_synsets[item]

        return synsets


def write_cls_names_file(output_filename, synsets):
    species = list(synsets.keys())

    with open(output_filename, 'w') as output_file:
        for item in species:
            output_file.write("%s\n" % item)


def define_database(dataset_folder, cls_names_file, tree_vs_nontree=True):
    if not os.path.exists(dataset_folder):
        raise Exception('Folder {} does not exist.'.format(dataset_folder))

    orig_img_exists = os.path.exists(os.path.join(dataset_folder, "original_images"))
    bb_exists = os.path.exists(os.path.join(dataset_folder, "bounding_boxes"))

    if not (orig_img_exists & bb_exists):
        raise Exception(
            'Folder {} must contain original_images/ and bounding_boxes/ subfolders.'.format(dataset_folder))

    my_imnetdb = ImnetDb(dataset_folder, tree_vs_nontree, shuffle=False, is_train=True,
                         names=os.path.join(dataset_folder, cls_names_file) )

    return my_imnetdb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="",  help='prefix which precedes the output io file')
    parser.add_argument('--folder', type=str, default="./", help='target folder to store the output io file')
    parser.add_argument('--augment-horizontally', type=bool, default=False, help='boolean to decide whether to augment horizontally the images')
    parser.add_argument('--show-sample', type=bool, default=False, help='boolean to show images from the dataset')
    parser.add_argument('--data', type=str, default=None, help='"all for including every class"')
    parsed_args = parser.parse_args()

    prefix = parsed_args.prefix
    target_folder = parsed_args.folder
    augment_horizontally = parsed_args.augment_horizontally

    default_classes = ['judas', 'palm', 'pine', 'fig', 'china tree']
    
    if parsed_args.data is None:
        target_classes = default_classes
    elif parsed_args.data == 'all':
        target_classes = 'all'
    else:
        raise RuntimeError("data option can either be None or 'all'") 
    
    synsets = select_synsets(target_classes)

    cls_names_file = 'imnet.names'
    write_cls_names_file(os.path.join(target_folder, cls_names_file), synsets)

    download_imnet_dataset(target_folder, synsets)

    tree_vs_nontree = True
    imnetdb = define_database(target_folder, cls_names_file, tree_vs_nontree)

    splitting = [0.8, 0.1, 0.1]
    str_list = imnetdb.split_imglist(splitting, root=None)

    print("saving lists to disk...")

    training_lst = os.path.join(target_folder, prefix + "train.lst")
    imnetdb.save_imglist(str_list[0], training_lst, shuffle=True)
    print("List file {} generated...".format(training_lst))

    # num_thread = args.num_thread
    num_thread = 1

    root_path = target_folder

    # this command shuffles the training dataset
    cmd_arguments = ["python",
                     os.path.join(curr_path, "im2rec.py"),
                     training_lst.replace('.lst', ''), os.path.abspath(root_path),
                     "--pack-label", "--num-thread", str(num_thread),
                     "--augment-horizontally", str(augment_horizontally)]

    subprocess.check_call(cmd_arguments)

    if splitting[1] > 0:
        validation_lst = os.path.join(target_folder, prefix + "validation.lst")

        imnetdb.save_imglist(str_list[1], validation_lst, shuffle=True)

        print("List file {} generated...".format(validation_lst))

        cmd_arguments = ["python",
                         os.path.join(curr_path, "im2rec.py"),
                         validation_lst.replace('.lst', ''), os.path.abspath(root_path),
                         "--pack-label", "--num-thread", str(num_thread)]
        subprocess.check_call(cmd_arguments)

    if splitting[2] > 0:
        testing_lst = os.path.join(target_folder, prefix + "test.lst")

        imnetdb.save_imglist(str_list[2], testing_lst, shuffle=False)
        print("List file {} generated...".format(testing_lst))

        cmd_arguments = ["python",
                         os.path.join(curr_path, "im2rec.py"),
                         testing_lst.replace('.lst', ''), os.path.abspath(root_path),
                         "--pack-label", "--num-thread", str(num_thread)]
        subprocess.check_call(cmd_arguments)

    print("Record files generated.")

    def show_sample_images():
        # test rec file
        import mxnet as mx

        record = mx.recordio.MXIndexedRecordIO(os.path.join(target_folder, prefix + 'train.idx'),
                                               os.path.join(target_folder, prefix + 'train.rec'), 'r')

        def plot_idx_record(idx):
            item = record.read_idx(idx)
            header, img = mx.recordio.unpack_img(item)

            print(header)
            import matplotlib.pyplot as plt

            plt.imshow(img)
            plt.show()

        plot_idx_record(0)

        num_images = 421
        if augment_horizontally:
            plot_idx_record(0+num_images)

        plot_idx_record(1)

        if augment_horizontally:
            plot_idx_record(1+num_images)

        plot_idx_record(2)

        if augment_horizontally:
            plot_idx_record(2+num_images)

        plot_idx_record(3)

        if augment_horizontally:
            plot_idx_record(3+num_images)

        plot_idx_record(4)

        if augment_horizontally:
            plot_idx_record(4+num_images)

    if parsed_args.show_sample:
        show_sample_images()

