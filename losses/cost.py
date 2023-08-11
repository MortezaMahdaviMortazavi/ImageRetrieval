import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .center_loss import CenterLoss
from .arcface import ArcFaceLoss

class CostApproximator(nn.Module):
    def __init__(self, num_classes, embedding_dim, lambda_c, margin=0.5, scale=30.0):
        super(CostApproximator, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(num_classes, embedding_dim)
        self.arcface_loss = ArcFaceLoss(embedding_dim, num_classes, margin, scale)

        self.center_weight = 1
        self.arcface_weight = 1
        self.cross_weight = 1
        
    def set_weights(self,weights):
        self.cross_entropy.weight = weights

    def weight_changing(self):
        self.center_weight+=0.01
        self.arcface_weigh+=0.01
        self.cross_weight-=0.01
        
    def forward(self, features, labels,predictions,step='train'):
        center_loss = self.center_loss(features, labels)
        # arcface_loss = self.arcface_loss(features, labels)
        ce_loss = self.cross_entropy(predictions,labels)
        _dict_ = {
            f"{step}_center_loss":center_loss,
            # f"{step}_arc_face":arcface_loss,
            f"{step}_cross_entropy":ce_loss
        }

        total_loss = ce_loss * self.cross_weight  +  center_loss * self.center_weight #+ arcface_loss * self.arcface_weight 

        return total_loss , _dict_
    


