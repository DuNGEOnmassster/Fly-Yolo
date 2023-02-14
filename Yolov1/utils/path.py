pathset = {
    "pathset1": {
        "root": r"E:\datasets\VOCdevkit2012\VOC2012",
        "Imageset_main_path": r"E:\datasets\VOCdevkit2012\VOC2012\ImageSets\Main",
        "train_path": r"E:\datasets\VOCdevkit2012\VOC2012\ImageSets\Main\train.txt",
        "test_path": r"E:\datasets\VOCdevkit2012\VOC2012\ImageSets\Main\test.txt",
        "val_path": r"E:\datasets\VOCdevkit2012\VOC2012\ImageSets\Main\val.txt",
    },

    "pathset2": {
        "root": "/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOC2012/",
        "Imageset_main_path": "/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOC2012/ImageSets/Main",
        "train_path": "/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOC2012/ImageSets/Main/train.txt",
        "test_path": "/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOC2012/ImageSets/Main/test.txt",
        "val_path": "/Volumes/NormanZ_980/Dataset/Object_Detection_Dataset/VOC2012/ImageSets/Main/val.txt",
    },
    
    "pathset3": {
        "root": "/home/zhengmingzhe/Super_Resolution/Reference/new/VOC2012",
        "Imageset_main_path": "/home/zhengmingzhe/Super_Resolution/Reference/new/VOC2012/ImageSets/Main",
        "train_path": "/home/zhengmingzhe/Super_Resolution/Reference/new/VOC2012/ImageSets/Main/train.txt",
        "test_path": "/home/zhengmingzhe/Super_Resolution/Reference/new/VOC2012/ImageSets/Main/test.txt",
        "val_path": "/home/zhengmingzhe/Super_Resolution/Reference/new/VOC2012/ImageSets/Main/val.txt",
    },
}

if __name__ == "__main__":
    tpath1 = pathset["pathset1"]["train_path"]
    print(tpath1)