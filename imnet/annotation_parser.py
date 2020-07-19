from xml.etree import ElementTree
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def extract_bounding_box(xml_file, names_wanted):
    '''
    The bounding box is defined by (xmin, ymin, xmax, ymax) coordinates.
    :param xml_file: The path to the xml_annotation that must be read
    :param names_wanted: The wnid(s) that you want to extract from an image.
    :return: (dict (wnid(str), list(tuple()) dictionary keyed by named wanted with a list of tuples containing each instance bounding box
    '''
    tree = ElementTree.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename")
    # print (filename.text)
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    depth = int(size.find("depth").text)
    # print (width, height, depth)
    objects = root.findall("object")
    returned = dict(zip(names_wanted, [[] for n in names_wanted]))

    for obj in objects:
        name = obj.find("name")
        if name.text in names_wanted:  # verify the object is in the list of objects we want to see
            b = obj.find("bndbox")
            xmin = int(b.find("xmin").text)
            ymin = int(b.find("ymin").text)
            xmax = int(b.find("xmax").text)
            ymax = int(b.find("ymax").text)
            returned[name.text] .append((xmin, ymin, xmax, ymax))
    return returned


def find_associated_annotation(anndir, impath):
    '''
    Given an image file, find the annotations for it following file naming conventions
    :param anndir: The directory where annotations are collected
    :param impath: The path of the image
    :return: annotation file path( (str) if it exists, else none
    '''
    imtitle = os.path.split(os.path.dirname(impath))[1]
    imname = os.path.splitext(os.path.basename(impath))[0]
    wnid = imname.split(sep="_")[0]
    retrieval_loc = os.path.join(anndir, imtitle, "Annotation", wnid)
    if os.path.exists(retrieval_loc):
        if imname + ".xml" in os.listdir(retrieval_loc):
            returned = os.path.join(retrieval_loc, imname + ".xml")
            if os.path.exists(returned):
                return returned
    else:
        return None

def draw_bounding_box_on_image(impath, bboxes):
    im =  Image.open(impath)
    plt.imshow(im)
    ax = plt.gca()

    for wnid, boxes in bboxes.items():
        for box in boxes:
            # comes as xmin, ymin, xmax, ymax
            # patches Rectangle determined in terms of bottom left coordinate
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    bbdir = os.path.join(os.path.dirname(os.getcwd()), "data", "imnet", "bounding_boxes")
    datadir = os.path.join(os.path.dirname(os.getcwd()), "data", "imnet", "original_images")

    names_wanted = ["n12513613"]
    count = 0
    for d,_,f in os.walk(datadir):
        for file in f:
            jpg = os.path.join(d, file)
            if os.path.splitext(jpg)[1] == ".JPEG":
                ann_xml = find_associated_annotation(bbdir, jpg)
                if ann_xml:
                    bboxes = extract_bounding_box(ann_xml, names_wanted)
                    draw_bounding_box_on_image(jpg, bboxes)
                    print (bboxes)
