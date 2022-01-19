#-*-coding:utf-8-*-
import pytest
from tests import _PATH_DATA
from src.data.load_mnist import mnist #beauty of having __init__.py files

import os.path
import torch
N_train, N_test = 5000, 25000
trainset, testset = mnist(_PATH_DATA+"/raw/", 1)

###############################
###  length of dataloaders  ### 
###############################
#@pytest.mark is honestly not necessary but I wanted to see
#how it worked
@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/raw/"), reason="Data files not found")
def test_on_dataloaders_len():
	assert len(trainset) == N_train
#(__len__ should be well defined for torch.DataLoaders)

#########################
###  size of samples  ###
#########################

#since the dataloader is created with shuffle = True
#the first n observations can be seen as a random 
#sample, so we simply check them
@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/raw/"), reason="Data files not found")
def sample_size_train():
	for _ in range(8):
		x = next(iter(trainset))
		assert x[0].shape == (1, 1, 28, 28), "should be a 4D tensor formatted as (1,1,28,28)"

@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"/raw/"), reason="Data files not found")
def sample_size_test():
	for _ in range(8):
		x = next(iter(testset))
		assert x[0].shape == (1, 1, 28, 28), "should be a 4D tensor formatted as (1,1,28,28)"

####################################
### check presence of all labels ###
####################################
def all_labels_present():
	for i in range(10): #labels are digits 0-9
		t = torch.tensor([11])
		idx = 0
		while t != torch.tensor([i]):
			t = next(iter(trainset))[1]
			idx += 1
		assert idx < N_train, "all labels should appear at least once in train set"
