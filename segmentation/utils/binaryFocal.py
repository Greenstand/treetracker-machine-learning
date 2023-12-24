import torch.nn as nn
import torch.nn.functional as F
import torch 

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for binary segmentation
    """
    def __init__(self, alpha=1, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        # Use BCEWithLogitsLoss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()