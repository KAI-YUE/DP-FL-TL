import os
import pickle
import numpy as np

from torchvision import transforms as T

def FashionMNIST(data_path):
    mean = 0.2861
    std = 0.3530

    channel = 1
    im_size = (28, 28)
    num_classes = 10

    with open(os.path.join(data_path, "train.dat"), "rb") as fp:
        dst_train = pickle.load(fp)
    
    with open(os.path.join(data_path, "test.dat"), "rb") as fp:
        dst_test = pickle.load(fp)

    # apply normalization
    train_images, test_images = dst_train["images"], dst_test["images"]
    train_images, test_images = train_images.astype(np.float32)/255, test_images.astype(np.float32)/255
    train_images, test_images = (train_images-mean)/std, (test_images-mean)/std

    # reshape images    
    train_images = train_images.reshape(-1, 1, 28, 28)
    test_images = test_images.reshape(-1, 1, 28, 28)

    dst_train["images"], dst_test["images"] = train_images, test_images

    properties = {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "dst_train": dst_train,
        "dst_test": dst_test,
    }

    class dataset_properties: pass
    for key, value in properties.items():
        setattr(dataset_properties, key, value)

    return dataset_properties
