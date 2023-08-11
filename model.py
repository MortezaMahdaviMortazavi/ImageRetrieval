import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import config
import pandas as pd

import torch.nn.functional as F
import math
import torchvision.models as models


# x.conv1
# x.bn1
# x.relu
# x.maxpool
# x.layer1
# x.layer2
# x.layer3
# x.layer4
# x.avgpool
# x.fc
class Discriminator(nn.Module):
    def __init__(self, num_classes,num_features=config.FEATURE_DIM):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        resnet50 = models.resnet50(pretrained=True)
        # self.resnet50.fc = nn.Identity()
        # Remove the last fully connected layer and the average pooling layer
        modules = list(resnet50.children())[:-2]
        self.resnet50_features = nn.Sequential(*modules)
        self.embed_layer = nn.Linear(2048, num_features)  # 2048 is the number of features from the ResNet-50 output
        self.fc = nn.Linear(num_features, num_classes)
        self.prelu = nn.PReLU()
        
        # self.attn_layer = SelfAttentionLayer(in_dim=num_features,hidden_dim=num_features//2)
    def make_grad(self,model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet50_features(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        _features = self.prelu(self.embed_layer(x))
        y = self.fc(F.dropout(_features,p=0.4))
        return  _features , y
    


class AttentionEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionEmbedding, self).__init__()
        self.embedding = nn.Linear(in_features=in_features, out_features=out_features)
        self.attention = nn.Linear(out_features, 1)  # Attention weights for each feature

    def forward(self, x):
        embedded = self.embedding(x)
        attention_weights = F.softmax(self.attention(embedded), dim=1)
        weighted_embedding = torch.mul(embedded, attention_weights)
        return weighted_embedding


# class Discriminator(nn.Module):
#     def __init__(self, num_classes,num_features=config.FEATURE_DIM):
#         super(Discriminator, self).__init__()
#         self.num_classes = num_classes
#         self.resnet = models.resnet50(pretrained=True)
#         self.make_grad(self.resnet.conv1,requires_grad=False)
#         self.make_grad(self.resnet.bn1,requires_grad=False)
#         self.make_grad(self.resnet.relu,requires_grad=False)
#         self.make_grad(self.resnet.maxpool,requires_grad=False)
#         self.make_grad(self.resnet.layer1,requires_grad=False)
#         self.make_grad(self.resnet.layer2,requires_grad=False)
#         self.make_grad(self.resnet.layer3,requires_grad=False)
#         self.make_grad(self.resnet.layer4,requires_grad=False)

#         self.embedding = AttentionEmbedding(in_features=2048,out_features=num_features)
#         self.fc = nn.Linear(in_features=num_features,out_features=num_classes)
#         self.dropout = nn.Dropout(p=0.5)
#         self.prelu = nn.PReLU()
        
#     def make_grad(self,model, requires_grad=True):
#         for param in model.parameters():
#             param.requires_grad = requires_grad

#     def forward(self, x):
#         with torch.no_grad():
#             x = self.resnet.conv1(x)
#             x = self.resnet.bn1(x)
#             x = self.resnet.relu(x)
#             x = self.resnet.maxpool(x)
#             x = self.resnet.layer1(x)
#             x = self.resnet.layer2(x)
#             x = self.resnet.layer3(x)
#             x = self.resnet.layer4(x)

#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)
#         # x = self.resnet.avgpool(x)
#         x = self.prelu(self.embedding(x))
#         y = self.fc(self.dropout(x))
#         return x,y


class EfficientNetV2(nn.Module):
    def __init__(self,num_classes,num_features=config.FEATURE_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features

        self.net = models.efficientnet_v2_l(pretrained=True)
        self.net.classifier = nn.Identity()
        self.embed_vec = nn.Linear(config.EFFICIENT_NETV2_IN_FEATURES,num_features)
        self.fc = nn.Linear(num_features,num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self,x):
        with torch.no_grad():
            x = self.net(x)

        _features = self.embed_vec(x)
        y = self.dropout(self.fc(F.relu(_features)))
        return _features , y
    

class CosineSimilarity(nn.Module):
    def __init__(self,eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, vec1, vec2):
        vec1 = vec1.unsqueeze(1)
        vec2 = vec2.unsqueeze(1)
        dot_product = torch.matmul(vec1, vec2.permute(0,2,1))
        norm_product = torch.norm(vec1, dim=-1)[:, None] * torch.norm(vec2, dim=-1)[:,None]
        return (dot_product / (norm_product + self.eps)).squeeze(1)







import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentionEmbedding, self).__init__()
        self.embedding = nn.Linear(in_features=in_features, out_features=out_features)
        self.attention = nn.Linear(out_features, 1)  # Attention weights for each feature

    def forward(self, x):
        embedded = self.embedding(x)
        attention_weights = F.softmax(self.attention(embedded), dim=1)
        weighted_embedding = torch.mul(embedded, attention_weights)
        return weighted_embedding

# # Example usage
# in_features = 2048
# out_features = 512

# embedding = AttentionEmbedding(in_features, out_features)

# # Assuming x is your input tensor
# x = torch.randn(32, in_features)

# output = embedding(x)
# print(output.shape)
