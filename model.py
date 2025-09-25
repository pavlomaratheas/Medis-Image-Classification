import torch.nn as nn
import torchvision.models as models


def get_model(n_classes=3, pretrained=True):
    "I adapt ResNet18 to my number of classes (3)"
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model
