import torch
import torch.nn as nn
import torch.optim as optim
import config
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import create_pair_dataloader
from losses import CostApproximator
from infer import evaluate_metrics
from sklearn.metrics import accuracy_score, f1_score
from utils import save_checkpoint , log_training_process , AverageMeter  # You need to define this utility function

class Trainer:
    def __init__(self, model, lr=0.001):
        self.model = model.to(config.DEVICE)
        self.lr = lr
        self.class_weights = None
        self.generate_class_weights(pd.read_csv(config.TRAIN_FILE))
        self.cost_function = CostApproximator(num_classes=82,embedding_dim=config.FEATURE_DIM,lambda_c=config.LAMBDA_C,margin=config.MARGIN,scale=config.SCALE).to(config.DEVICE)
        # self.cost_function.set_weights(self.class_weights)
        self.configure_optimizer()
        self.alpha = config.ALPHA
        self.data_dict = {}
        self.avg_meter = AverageMeter()
    
    def configure_optimizer(self):
        params = list(self.model.parameters()) + list(self.cost_function.parameters())
        self.optimizer = optim.AdamW(params,lr=0.001,weight_decay=0.001)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - epoch/(config.NUM_EPOCHS*10))

    def generate_class_weights(self,df):
        class_freq = df['label'].value_counts().sort_index()
        num_classes = len(class_freq)
        total_samples = len(df)
        class_weights = total_samples / (num_classes * class_freq)
        class_weights = torch.tensor(class_weights.values, dtype=torch.float)
        self.class_weights = class_weights.to(config.DEVICE)

    def forward(self, x):
        return self.model(x)


    def criterion(self,predictions,labels,features,step='train'):
        loss , losses_dict = self.cost_function(features,labels,predictions,step=step)
        return loss , losses_dict
    
    def loss_weight_decay(self):
        self.cost_function.weight_changing()

    def _common_step(self, batch, step):
        assert step in ['train', 'val']
        x, y = batch
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        feature_out, y_hat = self.forward(x)
        total_loss , losses = self.criterion(y_hat, y, feature_out,step)

        if step == 'train':
            self.optimizer.zero_grad()
            total_loss.backward()
            for param in self.cost_function.center_loss.parameters():
                param.grad.data *= (self.lr / (self.alpha * self.lr))
            self.optimizer.step()


        y_true_np = y.cpu().numpy()
        y_pred_np = y_hat.cpu().argmax(dim=1).numpy()
        accuracy = accuracy_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np,average='weighted')

        return total_loss, accuracy, f1 , losses
    
    def train(self, dataloader):
        self.model.train()

        result_dict = {
            "train_loss":0,
            "train_accuracy": 0,
            "train_f1_score": 0,
        }

        losses_dict = {}
        total_correct = 0
        total_samples = 0
        f1_scores = 0

        for x, y in tqdm(dataloader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            loss, accuracy, f1, all_losses = self._common_step((x, y), step='train')
            # Accumulate the losses from the current batch
            for loss_name, loss_value in all_losses.items():
                if loss_name in result_dict:
                    result_dict[loss_name] += loss_value
                else:
                    result_dict[loss_name] = loss_value
            result_dict['train_loss'] += loss
            total_correct += accuracy * x.size(0)
            total_samples += x.size(0)
            f1_scores += f1

        # Calculate average losses over the entire dataset
        num_batches = len(dataloader)
        if num_batches > 0:
            for loss_name, loss_value in losses_dict.items():
                result_dict[loss_name] = loss_value / num_batches

            result_dict["train_accuracy"] = total_correct / len(dataloader.dataset)
            result_dict["train_f1_score"] = f1_scores / num_batches
            result_dict['train_loss'] /= len(dataloader)

        return result_dict


    def evaluate(self, dataloader):
        self.model.eval()

        result_dict = {
            "Val Loss":0,
            "Val Acc": 0,
            "Val F1-Score": 0,
        }

        losses_dict = {}
        total_correct = 0
        total_samples = 0
        f1_scores = 0

        for x, y in tqdm(dataloader):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            total_loss, accuracy, f1, all_losses = self._common_step((x, y), step='val')
            for loss_name, loss_value in all_losses.items():
                    if loss_name in result_dict:
                        result_dict[loss_name] += loss_value
                    else:
                        result_dict[loss_name] = loss_value

            result_dict['Val Loss'] += total_loss
            total_correct += accuracy * x.size(0)
            total_samples += x.size(0)
            f1_scores += f1


        num_batches = len(dataloader)
        if num_batches > 0:
            for loss_name, loss_value in losses_dict.items():
                result_dict[loss_name] = loss_value / num_batches

            result_dict["Val Acc"] = total_correct / len(dataloader.dataset)
            result_dict["Val F1-Score"] = f1_scores / num_batches
            result_dict['Val Loss'] /= num_batches

        return result_dict

    
    def set_value(self,var,val):
        self.data_dict[var] = val
        
    def fit(self, trainloader, val_loader, num_epochs=config.NUM_EPOCHS):

        train_pair_dataloader , val_pair_dataloader, test_pair_dataloader = create_pair_dataloader(
            train_pair_df=pd.read_csv('Dataset/train_pairs.csv'),
            val_pair_df=pd.read_csv('Dataset/val_pairs.csv'),
            test_pair_df=pd.read_csv('Dataset/test_pairs.csv')
        )

        
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            train_dict = self.train(trainloader)  # Modify train function to return a dictionary
            self.scheduler.step() # adjust learning rate

            self.model.eval()  # Set model to evaluation mode
            val_dict = self.evaluate(val_loader)  # Modify evaluate function to return a dictionary

            result = f"Epoch {epoch + 1} | "
            for key, value in self.data_dict.items():
                result += f"{key}: {value:.4f} | "

            print(result)
            roc_auc, f1, precision, recall, accuracy, mean_cos_label_1, mean_cos_label_0 = evaluate_metrics(self.model, train_pair_dataloader)
            
            print(f"Train SimilarityDifference: {abs(mean_cos_label_1 - mean_cos_label_0):.4f} | Train Mean Cosine Similarity for being Similar: {mean_cos_label_1:.4f} | Train Mean Cosine Similarity for not being similar: {mean_cos_label_0:.4f} | Train ROC-AUC-Score: {roc_auc:.4f} | Train Data_Pair_F1-Score: {f1:.4f} | Train Data_Pair_Precision: {precision:.4f} | Data_Pair_Recall: {recall:.4f} |Train Data_Pair_Accuracy: {accuracy:.4f}")


            roc_auc, f1, precision, recall, accuracy, mean_cos_label_1, mean_cos_label_0 = evaluate_metrics(self.model, val_pair_dataloader)
            
            print(f" Val SimilarityDifference: {abs(mean_cos_label_1 - mean_cos_label_0):.4f} | Val Mean Cosine Similarity for being Similar: {mean_cos_label_1:.4f} | Val Mean Cosine Similarity for not being similar: {mean_cos_label_0:.4f} | Val ROC-AUC-Score: {roc_auc:.4f} | Val Data_Pair_F1-Score: {f1:.4f} | Val Data_Pair_Precision: {precision:.4f} | Val Data_Pair_Recall: {recall:.4f} | Val Data_Pair_Accuracy: {accuracy:.4f}")



            # Update the metrics to the self.avg_meter and self.data_dict dictionaries
            self.data_dict.update(train_dict)
            self.data_dict.update(val_dict)
            self.data_dict['val_pair_f1'] = f1
            self.data_dict['val_pair_precision'] = precision
            self.data_dict['val_pair_recall'] = recall
            self.data_dict['val_pair_accuracy'] = accuracy
            self.data_dict['val_pair_roc_auc'] = roc_auc
            self.data_dict['val_is_similar'] = mean_cos_label_1
            self.data_dict['val_not_similar'] = mean_cos_label_0

            # Construct the result string

            log_training_process(result=result)

            # Save checkpoint after each epoch
            checkpoint_path = config.CHECKPOINT_FILE
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                epoch=epoch
            )
        # self.loss_weight_decay()

        print("Training finished.")

        test_roc_auc, test_f1, test_precision, test_recall, test_accuracy, test_mean_cos_sim_label_1, test_mean_cos_sim_label_0 = evaluate_metrics(self.model, test_pair_dataloader)

        print("Testset: ROC AUC:", test_roc_auc)
        print("Testset: F1 Score:", test_f1)
        print("Testset: Precision:", test_precision)
        print("Testset: Recall:", test_recall)
        print("Testset: Accuracy:", test_accuracy)
        print("Testset: Mean Cosine Similarity for label 1:", test_mean_cos_sim_label_1)
        print("Testset: Mean Cosine Similarity for label 0:", test_mean_cos_sim_label_0)

        return self.avg_meter
