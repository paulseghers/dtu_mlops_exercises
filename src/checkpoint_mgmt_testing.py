# does exactly the same as checkpoint_mgmt but with different import path 
# to resolve import issues when importing these methods from tests/ dir
# also doesn't print error messages as pytest handles those

import torch
from src.model import MyAwesomeModel, mnist_classifier

def save_checkpoint(filepath, model):
    if isinstance(model, MyAwesomeModel):
        checkpoint = {
            'height': model.height,
            'width': model.width,
            'channels': model.channels,
            'classes': model.classes,
            'dropout': model.dropout_rate,
            'state_dict': model.state_dict()}
    else:
        checkpoint = model.state_dict()
    torch.save(checkpoint, filepath)
    print("saved model successfully at {}".format(filepath))

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    try:
        model = MyAwesomeModel(
            height=checkpoint['height'],
            width=checkpoint['width'],
            channels=checkpoint['channels'],
            classes=checkpoint['classes'],
            dropout=checkpoint['dropout'])
    except KeyError:
        model = mnist_classifier()
    model = mnist_classifier()
    model.load_state_dict(checkpoint)#in this case its simply the state dict
    return model
