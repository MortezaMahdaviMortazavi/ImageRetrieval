import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=70.0):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, target):
        # x_dim = [batch_size,embed_dim]
        # Normalize feature embeddings
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cos_theta = F.linear(x_norm, w_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # Compute the arc margin
        target_one_hot = torch.zeros(cos_theta.size(), device=x.device)
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        
        sin_theta = torch.sqrt(1.0 - cos_theta**2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Apply the margin only to the correct class
        cos_theta_m[target_one_hot.bool()] = cos_theta[target_one_hot.bool()] - self.mm
        
        # Scale the embeddings
        logits = cos_theta_m * self.scale
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, target)
        
        return loss