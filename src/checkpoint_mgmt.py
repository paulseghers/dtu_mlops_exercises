import torch
from model import MyAwesomeModel, mnist_classifier


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
    print("entered here \n \n")
    try:
        model = MyAwesomeModel(
            height=checkpoint['height'],
            width=checkpoint['width'],
            channels=checkpoint['channels'],
            classes=checkpoint['classes'],
            dropout=checkpoint['dropout'])
    except KeyError:
        print("error loading the attributes :/")
        print("you're probably trying to load a CNN from a savefile of a Linear NN\n attempting to fix for you...\n")
        model = mnist_classifier()
    model = mnist_classifier()
    model.load_state_dict(checkpoint)#in this case its simply the state dict
    print("loaded model successfully from {}".format(filepath))
    return model
