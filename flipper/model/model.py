import torchvision
from torch import nn

def pretrained_resnet18():
    """ Returns pre-trained resnet18 with untrained final fc layer """
    model = torchvision.models.resnet18(pretrained=True)

    # Fix parameters of all pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model