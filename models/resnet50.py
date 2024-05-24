import torch
import torch.nn as nn
import torchvision.models as models


class CustomResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last layers
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Modify the last layer for binary classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        """
        :param x: Tensor with shape (batch, number of images, channels, H, W)
        :return:
        """
        orig_x = torch.clone(x)
        batch_size = x.shape[0]
        n_patches = x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[-2], x.shape[-1])
        outputs = self.resnet(x)
        preds = torch.sigmoid(outputs)
        preds = preds.view(batch_size, n_patches, -1)
        outputs = preds.mean(dim=1)
        return outputs
