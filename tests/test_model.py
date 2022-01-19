#-*-coding:utf-8-*-
import torch
import pytest
from tests import _PATH_MODELS
from src.model import MyAwesomeModel, mnist_classifier
from src.checkpoint_mgmt_testing import load_checkpoint #beauty of having __init__.py files

model = load_checkpoint(_PATH_MODELS+"/mnist_classifier.pt")

def test_error_on_wrong_shape():
   with pytest.raises(RuntimeError):
      model(torch.randn(1,2,3))
