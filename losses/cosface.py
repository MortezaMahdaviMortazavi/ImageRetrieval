import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LMCLLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, s=30.0, m=0.40,use_gpu=True):
        super(LMCLLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s
        self.m = m

        # Create a learnable parameter for the class centers (weights)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))#.to(config.DEVICE)
        # if use_gpu:
        #     self.centers = self.centers.cuda()

        nn.init.xavier_normal_(self.centers, gain=1)

    def forward(self, features, targets):
        """
        Compute the LMCL loss.

        Parameters:
            features: Tensor of shape (batch_size, feat_dim) representing the input features.
            targets: Tensor of shape (batch_size,) containing the class labels (0 to num_classes-1).

        Returns:
            loss: LMCL loss value.
        """
        # Normalize the features and the class centers
        features = F.normalize(features, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)

        # Get the center for each sample using the target labels
        centers_batch = centers[targets]

        # Compute the cosine similarity between features and centers
        cos_sim = torch.matmul(features, centers_batch.t())

        # Compute the logits by scaling the cosine similarity
        logits = self.s * cos_sim
        
        # Add margin to the correct class
        logits_target = logits.gather(1, targets.unsqueeze(1))
        logits_target = logits_target.view(-1)

        # Compute the LMCL loss
        loss = F.cross_entropy(logits - self.m, targets)

        return loss