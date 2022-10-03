''' transform various annotation format into json '''

import numpy as np
import os
import cv2
import json
import xml.etree.ElementTree as ET
import argparse
import yaml

f_coco = open(r"E:\datasets\COCO\annotations\instances_val2017.json", 'r')
json_file = json.load(f_coco)

for item in json_file:
    print(item)

images = json_file["images"]
annotations = json_file["annotations"]
categories = json_file["categories"]

f_class = open("./VOC_names.yaml", 'r')
class_names = yaml.load(f_class, Loader=yaml.SafeLoader)

def args():
    args = argparse.ArgumentParser()
    args.add_argument("--output", default=r"E:\datasets\yolo_dataset\VOC2007\ImageSets\Main", help="train, val, test file store path")
    opt = args.parse_args()
    return opt

def get_img_information(img_path):
    # images
    full_name = os.path.split(img_path)[1]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return [full_name, img_path, h, w]

def get_anno_information(anno_path):
    # annotation
    target = ET.parse(anno_path)
    root = target.getroot()

    labels = []
    for object in root.iter("object"):
        difficult = int(object.find("difficult").text)
        class_name = object.find("name").text
        class_ind = class_names["class_names"].index(class_name)

        bndbox = object.find("bndbox")
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        points = []
        for pt in pts:
            point = float(bndbox.find(pt).text)
            points.append(point)

        box = [0, 0, 0, 0]
        box[0] = (points[0] + points[2]) / 2
        box[1] = (points[1] + points[3]) / 2
        box[2] = np.abs(points[0] - points[2])
        box[3] = np.abs(points[1] - points[3])
        label = [difficult] + [class_ind] + box
        labels.append(label)

    labels = np.array(labels, dtype=np.float32)
    return labels

if __name__ == "__main__":

    opt = args()

    root = r"E:\datasets\yolo_dataset\VOC2007"
    img_root = os.path.join(root, "JPEGImages")
    anno_root = os.path.join(root, "Annotations")

    img_paths = [os.path.join(img_root, i) for i in os.listdir(img_root)]
    anno_paths = [os.path.join(anno_root, i) for i in os.listdir(anno_root)]

    n = len(img_paths)

    imgs = []
    ys = []
    for i in range(n):
        img_infor = get_img_information(img_paths[i])
        labels = get_anno_information(anno_paths[i])
        img_infor += [i]
        imgs.append(img_infor)
        flag = np.repeat([[i]], labels.shape[0], axis=0)
        y = np.concatenate((labels, flag), axis=1).tolist()
        ys.append(y)

    categories = {}
    for i, class_name in enumerate(class_names["class_names"]):
        categories[i] = class_name

    with open(os.path.join(opt.output, "output.json"), 'w') as f:
        str1 = '"images": ['
        f.write(str1)
    with open(os.path.join(opt.output, "output.json"), 'a') as f:
        for img in imgs:
            full_name = img[0]
            img_path = img[1]
            h = img[2]
            w = img[3]
            img_id = img[4]
            write_str = {}
            write_str["file_name"] = str(full_name)
            write_str["coco_url"] = str(img_path)
            write_str["height"] = str(h)
            write_str["width"] = str(w)
            write_str["id"] = str(img_id)
            js = json.dumps(write_str)
            f.write(js)
        f.write('],')

    with open(os.path.join(opt.output, "output.json"), 'a') as f:
        str1 = '"annotations": ['
        f.write(str1)

    with open(os.path.join(opt.output, "output.json"), 'a') as f:
        id = 0
        for i, y in enumerate(ys):
            for item in y:
                full_name = img[0]
                img_path = img[1]
                h = img[2]
                w = img[3]
                img_id = img[4]
                write_str = {}
                write_str["iscrowed"] = str(item[0])
                write_str["image_id"] = str(item[6])
                write_str["bbox"] = str(item[2:6])
                write_str["category_id"] = str(item[1])
                write_str["id"] = str(id)
                id += 1
                js = json.dumps(write_str)
                f.write(js)
        f.write('],')

    with open(os.path.join(opt.output, "output.json"), 'a') as f:
        str1 = '"categories": ['
        f.write(str1)
    with open(os.path.join(opt.output, "output.json"), 'a') as f:
        js = json.dumps(categories)
        f.write(js)
        f.write(']')











