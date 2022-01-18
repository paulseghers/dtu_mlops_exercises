import torch
import numpy as np
import os.path
from torch.utils.data import DataLoader, TensorDataset
import torchvision

import glob
import os
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset


def mnist(filepath, batch_size):
    #data_path = "data/raw/"
    train_data_path = filepath + "/train_merged.npz"
    test_data_path = filepath + "/test.npz"
    
    train_blob = np.load(train_data_path)
    test_blob = np.load(test_data_path)
    ### train set ###
    
    train_images = torch.tensor(train_blob["images"]).view(5000, 1, 28, 28)
    train_labels = torch.tensor(train_blob["labels"])
    train_data = []
    for i in range(len(train_images)):
        train_data.append([train_images[i], train_labels[i]])

    trainloader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)


    ### test set ###
    test_data = []
    
    test_images = torch.tensor(test_blob["images"]).view(5000, 1, 28, 28)
    test_labels = torch.tensor(test_blob["labels"])
    test_data = []
    for i in range(len(test_images)):
        test_data.append([test_images[i], test_labels[i]])

    testloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

    print("testloadr type: ", testloader)
    return trainloader, testloader
"""
def mnist():
    print("running data script...")
    #looking at the directory structure, we have npz 5 files for training:
    filenames = ["".join(("../../../data/corruptmnist/train_",str(i),".npz")) for i in range(5)]
    
    merged_data = {}
    for data in data_all:
        [merged_data.update({k: v}) for k, v in data.items()]
    np.savez(data_path + "train/train_merged.npz", **merged_data)

    debug_test = True
    debug_train= True
    train, test = None, None #for debugging
    # training set#
    train_data=[]
    if debug_train: #my computer is very smol :(
        data = np.load(filenames[0])
        dat, lbl = data['images'], data['labels']
        train_data+=[ [ np.array(dat[i])[None,:,:], lbl[i] ] for i in range(len(dat))]
    #[None,:,:] is to match dimension of my Conv2D layer, I still don't 
    #know why the dataloader (or network layer) flips dimensions like this
    #print("shape: ",np.array(train_data)[0][0].shape)   

    # test set #
    test_data=[]
    if debug_test:
        testdata_npz = np.load("../../../data/corruptmnist/test.npz")
        testdat, testlbl = testdata_npz['images'], testdata_npz['labels']    
        test_data+=[ [ np.array(testdat[i])[None,:,:], testlbl[i] ] for i in range(len(testdat))]
        #print("len testdat: ",len(testdat), "\n testlbl: ", len(testlbl))

    # creating a list like this means __getitem__ and __len__ methods propagate; 
    # so we can shove this into a dataloader
    if debug_train:
        train = DataLoader(train_data, shuffle=True, batch_size=64)
    if debug_test:
        test = DataLoader(test_data, shuffle=True, batch_size=64)

    print("\n everything ran in data\n")

    return train, test
"""