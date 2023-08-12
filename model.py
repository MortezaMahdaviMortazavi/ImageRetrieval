<<<<<<< HEAD
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
        resnet101 = models.resnet101(pretrained=True)
        # self.resnet50.fc = nn.Identity()
        # Remove the last fully connected layer and the average pooling layer
        modules = list(resnet101.children())[:-2]
        self.resnet101_features = nn.Sequential(*modules)
        self.embed_layer = nn.Linear(2048, num_features)  # 2048 is the number of features from the ResNet-50 output
        self.fc = nn.Linear(num_features, num_classes)
        self.prelu = nn.PReLU()
        
        # self.attn_layer = SelfAttentionLayer(in_dim=num_features,hidden_dim=num_features//2)
    def make_grad(self,model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet101_features(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        _features = self.prelu(self.embed_layer(x))
        y = self.fc(F.dropout(_features,p=0.4))
        return  _features , y
    


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



class HybridCNNTransformer(nn.Module):
    def __init__(self,num_classes=82):
        super().__init__()
        self.swin_feature_dim = 1024
        self.resnet_feature_dim = 8192

        self.swin_transformer = models.swin_b(pretrained=True)
        self.resnet = models.resnet101(pretrained=True)

        self.swin_feature_extractor = nn.Sequential(*list(self.swin_transformer.children())[:-1])
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.embedd_layer = nn.Linear(self.swin_feature_dim + self.resnet_feature_dim , config.FEATURE_DIM)
        self.fc = nn.Linear(config.FEATURE_DIM,num_classes)

        self.prelu = nn.PReLU(num_parameters=1,init=0.2)

    def forward(self,x):
        with torch.no_grad():
            swin_out = self.swin_feature_extractor(x)
            resnet_out = self.resnet_feature_extractor(x)
            resnet_out = F.adaptive_avg_pool2d(resnet_out, (2, 2))
            resnet_out = torch.flatten(resnet_out, 1)

        freeze_features = torch.cat([swin_out,resnet_out],dim=-1)

        embed_features = self.prelu(self.embedd_layer(freeze_features))
        classify = self.fc(embed_features)

        return embed_features , classify



    


