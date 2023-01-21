# -*- coding: utf-8 -*-

import os
import shutil

dataset_path = "/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOCdevkit/VOC2012/"

ann_filepath = dataset_path + '/Annotations/'#origianl dataset directory
img_filepath = dataset_path + '/JPEGImages/'#origianl dataset directory
img_savepath = dataset_path + '/JPEGImages/'#creating new dataset directory
ann_savepath = dataset_path + '/Annotations/'#creating new dataset directory

if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)
 
if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)

names = locals()
# classes = ['aeroplane','bicycle','bird', 'boat', 'bottle',
           # 'bus', 'car', 'cat', 'chair', 'cow','diningtable',
           # 'dog', 'horse', 'motorbike', 'pottedplant',
           # 'sheep', 'sofa', 'train', 'tvmonitor', 'person']
classes =['person']#select what categories you want.
 
for file in os.listdir(ann_filepath):
    print(file)
    c1 = 0
    c2 = 1
    fp = open(ann_filepath + file)
    ann_savefile=ann_savepath+file
    fp_w = open(ann_savefile, 'w')
    lines = fp.readlines()

    ind_start = []
    ind_end = []
    ind_mid=[]
    lines_id_start = lines[:]
    lines_id_end = lines[:]
    lines_id_mid = lines[:]
    if "\t<segmented>0</segmented>\n" in lines_id_start:
        c1 = lines_id_start.index("\t<segmented>0</segmented>\n")
    if "\t<segmented>1</segmented>\n" in lines_id_start:
        c2 = lines_id_start.index("\t<segmented>1</segmented>\n")
    d = lines_id_start.index("\t</source>\n")

    # if '\t</size>\n' in lines[:]:
        # print(lines[:])
    # classes1 = '\t\t<name>bicycle</name>\n'
    # classes2 = '\t\t<name>motorbike</name>\n'
    # classes3 = '\t\t<name>bus</name>\n'
    # classes4 = '\t\t<name>car</name>\n'
    classes5 = '\t\t<name>person</name>\n'#select what categories you want.

    #Find the 'object' block in the XML file and record it
    while "\t<object>\n" in lines_id_start:
        a = lines_id_start.index("\t<object>\n")
        ind_start.append(a)
        lines_id_start[a] = "delete"


    while "\t</object>\n" in lines_id_end:
        b = lines_id_end.index("\t</object>\n")
        ind_end.append(b)
        lines_id_end[b] = "delete"
    # print('ind_end',ind_end)
    #'names' holds all 'object' blocks
    i = 0
    for k in range(0, len(ind_start)):
        names['block%d' % k] = []
        for j in range(0, len(classes)):
            if classes[j] in lines[ind_start[i] + 1]:
                a = ind_start[i]
                for o in range(ind_end[i] - ind_start[i] + 1):
                    names['block%d' % k].append(lines[a + o])
                break
        i += 1

    #The front information of the XML file
    string_start = lines[0:ind_start[0]]
    #The back  information of the XML file
    string_end = [lines[len(lines) - 1]]
    #Search within the given class and, if present, write 'object' block information
    a = 0
    for k in range(0, len(ind_start)):
        # if classes1 in names['block%d' % k]:
            # a += 1
            # string_start += names['block%d' % k]
        # if classes2 in names['block%d' % k]:
            # a += 1
            # string_start += names['block%d' % k]
        # if classes3 in names['block%d' % k]:
            # a += 1
            # string_start += names['block%d' % k]
        # if classes4 in names['block%d' % k]:
            # a += 1
            # string_start += names['block%d' % k]
        if classes5 in names['block%d' % k]:
            a += 1
            string_start += names['block%d' % k]  #select what categories you want.
    # for str in string_mid:
        # string_start+=str
    if c1 != 0&a<d:
        string_start += lines[c1:d+1] 
    else:
        if c2 != 0&a<d:
            string_start += lines[c2:d+1] 

    string_start += string_end
    for c in range(0, len(string_start)):
        fp_w.write(string_start[c])

    fp_w.close()
    #If there is no module we are looking for, delete this xml, and copy the picture if there is
    if a == 0:
        os.remove(ann_savepath+file)
    else:
        name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"
        shutil.copy(name_img, img_savepath)
    fp.close()