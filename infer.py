# from model import CosineSimilarity,Discriminator
from dataloader import FlowerPairDataset
from tqdm import tqdm

import config

import torch
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score


def evaluate_metrics(model, dataloader, device='cuda'):
    model.eval()
    true_labels = []
    predicted_probs = []
    cos_sim_label_1 = []
    cos_sim_label_0 = []

    with torch.no_grad():
        for x1, x2, labels in tqdm(dataloader):
            x1, x2 = x1.to(config.DEVICE), x2.to(config.DEVICE)

            # Forward pass through the model
            feature_out1, y_hat1 = model(x1)
            feature_out2, y_hat2 = model(x2)

            similarities = torch.nn.functional.cosine_similarity(feature_out1, feature_out2)
            predicted_probs.extend(similarities.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                if labels[i] == 1:
                    cos_sim_label_1.append(similarities[i].item())
                else:
                    cos_sim_label_0.append(similarities[i].item())

    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)
    normalized_probs = (predicted_probs + 1) / 2

    threshold = 0.4
    predicted_labels = (predicted_probs > threshold).astype(int)
    roc_auc = roc_auc_score(true_labels, normalized_probs)
    f1 = f1_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels,predicted_labels)

    mean_cos_sim_label_1 = np.mean(cos_sim_label_1)
    mean_cos_sim_label_0 = np.mean(cos_sim_label_0)

    return roc_auc , f1 , precision , recall , accuracy , mean_cos_sim_label_1 , mean_cos_sim_label_0


import torchvision.models as models
class Dis(nn.Module):
    def __init__(self, num_classes,num_features=config.FEATURE_DIM):
        super(Dis, self).__init__()
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
        # _features = self.prelu(self.embed_layer(x))
        # y = self.fc(F.dropout(_features,p=0.4))
        return  x , x
    

# # import utils
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# model = Dis(num_classes=82,num_features=2048)
# df = pd.read_csv('Dataset/test_pairs.csv')
# dataset = FlowerPairDataset(df=df,transform=transform)
# loader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False)
# # model = Discriminator(num_classes=82,num_features=2048).to(config.DEVICE)
# # utils.load_checkpoint(checkpoint_path='logs/ResnetWithArcCenter.pt',model=model)
# print(evaluate_metrics(model=model,dataloader=loader))
