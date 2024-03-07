import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Loss(nn.Module):
    def __init__(self, ac_model, reduction: str = 'mean'):
        """The auxiliary classifier loss which is intended to be used to ensure that underlying abnormality patterns are
        preserved during the anonymization process.

        :param ac_model: nn.Module
            A pre-trained abnormality classifier (DenseNet-121).
        :param reduction: str
            Loss reduction method: default value is 'mean'; other options are 'sum' or 'none'.
        """

        super().__init__()
        self.ac_model = ac_model

        # Set model to evaluation mode
        self.ac_model.eval()

        # Turn on gradient computation
        for param in self.ac_model.parameters():
            param.requires_grad = True

        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()

    def forward(self, deformed_image, target_labels):
        # The abnormality classification model was trained with 3-channel inputs
        # --> expand tensors to have 3 identical channels
        deformed_image = deformed_image.expand(-1, 3, -1, -1)

        # Apply the ImageNet transform (since the classifier was trained with the ImageNet transform as well)
        resize = transforms.Resize(224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        deformed_image = normalize(resize(deformed_image))
        output = self.ac_model(deformed_image)
        
        target_labels=target_labels.t()
        target_tensor=[]
        for i in target_labels:
            target_tensor.append(i)
        
        loss_list = []
        for i in range(len(target_tensor)):
            loss = self.ce_loss(output[i], target_tensor[i].long())
            loss_list.append(loss)

        average_loss = torch.tensor(torch.stack(loss_list).mean().item())

        return average_loss
