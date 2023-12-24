import torch.nn as nn
import torch.nn.functional as F
import torch 


class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation
    """
    def __init__(self, smooth=1.0):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = torch.sigmoid(inputs)  # Apply sigmoid to squash outputs between 0 and 1
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # Calculate Dice loss
        dice_loss = 1 - dice
        return dice_loss