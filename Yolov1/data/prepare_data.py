''' divide the data into train_data, validate_data and test_data '''

import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.path import pathset

random.seed(42)

# 数据集路径
chosen_pathset = "pathset2"
root = pathset[chosen_pathset]["root"]

# 训练集 + 验证集 : 测试集 = 0.8
train_val = 0.8
# 训练集: 验证集 = 0.8
train = 0.8

img_supported = ['.png', '.jpg']
anno_supported = ['.xml']

def divide_dataset(img_names):

    f_train = open(os.path.join(main_root, 'train.txt'), 'w')
    f_test = open(os.path.join(main_root, 'test.txt'), 'w')
    f_val = open(os.path.join(main_root, 'val.txt'), 'w')
    f_train_val = open(os.path.join(main_root, 'train_val.txt'), 'w')

    # 获得测试集
    train_val_names = random.sample(img_names, int(len(img_names) * train_val))
    for img_name in img_names:
        if img_name not in train_val_names:
            f_test.write(img_name+'\n')
        else:
            f_train_val.write(img_name+'\n')

    # 获得训练集和验证集
    train_names = random.sample(train_val_names, int(len(train_val_names) * train))
    for img_name in train_val_names:
        if img_name in train_names:
            f_train.write(img_name+'\n')
        else:
            f_val.write(img_name+'\n')

    f_train.close()
    f_val.close()
    f_test.close()

if __name__ == "__main__":
    # 生成图像目录
    img_root = os.path.join(root, 'JPEGImages')
    # 生成标注文件目录
    anno_root = os.path.join(root, 'Annotations')
    # 生成训练集+验证集+测试集目录
    main_root = os.path.join(root, 'ImageSets', 'Main')
    assert img_root, "图像目录不存在"
    assert anno_root, "标注文件目录不存在"
    assert main_root, "存在训练集+验证集+测试集目录不存在"

    # 判断图像与标注文件是否数量一致
    img_paths = [os.path.join(img_root, i) for i in os.listdir(img_root) if os.path.splitext(i)[-1] in img_supported]
    anno_paths = [os.path.join(anno_root, i) for i in os.listdir(anno_root) if os.path.splitext(i)[-1] in anno_supported]
    assert len(img_paths) == len(anno_paths), "图像数据和标注文件数据数量不一致"

    # 保存图像的格式, 比如'.png', '.jpg'
    f_postfix = open(os.path.join(main_root, 'postfix.txt'), 'w')
    f_postfix.write(os.path.splitext(img_paths[0])[-1])
    f_postfix.close()

    # 获得图像名称, 用于获得训练集+验证集+测试集
    img_names = [os.path.basename(os.path.splitext(i)[0]) for i in img_paths]

    divide_dataset(img_names)