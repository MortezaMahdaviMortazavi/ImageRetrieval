<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> ca7d9da84a74cc9f1a85f0eb812d853e11a9db46
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
<<<<<<< HEAD
=======
=======
import torch
import torch.nn as nn
import torch.optim as optim
import config
import pandas as pd
import numpy as np

from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

from loss.center_loss import CenterLoss
from loss.cosface import LMCLLoss , LMCL_loss
from loss.circle_loss import CircleLoss

from infer import evaluate_metrics , create_pair_dataloader, evaluate_metrics_knn
from sklearn.metrics import accuracy_score, f1_score
from utils import save_checkpoint , log_training_process , AverageMeter  # You need to define this utility function

class Trainer:
    def __init__(self, model, lr=0.001):
        self.model = model.to(config.DEVICE)
        self.lr = lr
        self.class_weights = None
        self.generate_class_weights(pd.read_csv(config.TRAIN_FILE))

        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weights)
        self.center_loss = CenterLoss(num_classes=82, feat_dim=config.FEATURE_DIM, use_gpu=True)
        self.lmcl_loss = LMCL_loss(num_classes=82,feat_dim=config.FEATURE_DIM,use_gpu=True)

        self.configure_optimizer()
        self.circle_loss = CircleLoss(m=0.25,gamma=config.BATCH_SIZE*2)
        self.alpha = config.ALPHA
        self.data_dict = {}
        self.avg_meter = AverageMeter()
    
    def configure_optimizer(self):
        params = list(self.model.parameters()) + list(self.center_loss.parameters())
        self.optimizer = optim.Adam(params,lr=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def generate_class_weights(self,df):
        class_freq = df['label'].value_counts().sort_index()
        num_classes = len(class_freq)
        total_samples = len(df)
        class_weights = total_samples / (num_classes * class_freq)
        class_weights = torch.tensor(class_weights.values, dtype=torch.float)
        self.class_weights = class_weights.to(config.DEVICE)

    def forward(self, x):
        return self.model(x)
    
    def set_methods(self,use_entropy=True,use_center=True,use_lmcl=True,):
        self.use_center = use_center
        self.use_entropy = use_entropy
        self.use_lmcl = use_lmcl

    def criterion(self, y_hat, y, features):
        losses = {}
        total_loss = 0.0

        if self.use_entropy:
            cross_entropy = self.cross_entropy(y_hat, y)
            total_loss += cross_entropy
            losses["CrossEntropy"] = cross_entropy
        else:
            losses['CrossEntropy'] = "Not_Using"

        if self.use_lmcl:
            lmcl_loss = self.lmcl_loss(features, y)
            total_loss += lmcl_loss
            losses["LMCLLoss"] = lmcl_loss
        
        else:
            losses['LMCLLoss'] = "Not_Using"

        if self.use_center:
            center_loss = self.center_loss(features, y)
            total_loss += center_loss * self.alpha
            losses["CenterLoss"] = center_loss
        else:
            losses["CenterLoss"] = "Not_Using"

        losses["TotalLoss"] = total_loss.item()
        return total_loss, losses

    def _common_step(self, batch, step):
        assert step in ['train', 'val']
        x, y = batch
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        feature_out, y_hat = self.forward(x)
        total_loss , losses = self.criterion(y_hat, y, feature_out)

        if step == 'train':
            self.optimizer.zero_grad()
            total_loss.backward()
            for param in self.center_loss.parameters():
                param.grad.data *= (self.lr / (self.alpha * self.lr))
            self.optimizer.step()

        torch.save(self.center_loss.state_dict(),'checkpoints/center_loss_statedicts.pt')
        torch.save(self.lmcl_loss.state_dict(),'checkpoints/lmcl_loss_statedicts.pt')
        # Convert PyTorch tensors to NumPy arrays
        y_true_np = y.cpu().numpy()
        y_pred_np = y_hat.cpu().argmax(dim=1).numpy()
        accuracy = accuracy_score(y_true_np, y_pred_np)
        f1 = f1_score(y_true_np, y_pred_np,average='weighted')

        return total_loss, accuracy, f1 , losses['CenterLoss'] , losses['LMCLLoss'] , losses['CrossEntropy']
    
    def train(self, dataloader):
        self.model.train()
        result = ""
        losses = 0
        total_correct = 0
        total_samples = 0
        f1_scores = 0
        
        center_avg = 0
        lmcl_avg = 0
        cross_entropy_avg = 0

        for x, y in tqdm(dataloader):
            x , y = x.to(config.DEVICE) , y.to(config.DEVICE)
            loss, accuracy, f1 , center_loss , lmcl_loss , cross_entropy = self._common_step((x, y), step='train')
            losses += loss.item()

            if self.use_center:
                center_avg += center_loss.item()
            else:
                center_avg = center_loss

            if self.use_lmcl:
                lmcl_avg += lmcl_loss.item()
            else:
                lmcl_avg = lmcl_loss

            if self.use_entropy:
                cross_entropy_avg += cross_entropy
            else:
                cross_entropy_avg = cross_entropy

            total_correct += accuracy * x.size(0)
            total_samples += x.size(0)
            f1_scores += f1

        try:
            train_loss_avg = losses / len(dataloader)
            result += f"Train Loss: {train_loss_avg:.4f} | "
        except:
            pass
        
        try:
            center_avg = center_avg / len(dataloader)
            result += f"Train Center Loss: {center_avg:.4f} | "
        except:
            # its not a number so we eliminate .4f
            result += f"Train Center Loss: {center_avg} | "
        
        try:
            lmcl_avg = lmcl_avg / len(dataloader)
            result += f"Train lmcl Loss: {lmcl_avg:.4f} | "
        except:
            result += f"Train lmcl Loss: {lmcl_avg} | "

        try:
            cross_entropy_avg = cross_entropy_avg / len(dataloader)
            train_acc_avg = total_correct / total_samples
            train_f1_avg = f1_scores / len(dataloader)
            result += f"Train Cross_entropy Loss: {cross_entropy_avg:.4f} | Train Acc: {train_acc_avg:.4f} | Train F1-Score: {train_f1_avg:.4f}"
        except:
            result += f"Train Cross_entropy Loss: {cross_entropy_avg}"

        print(result)

        return train_loss_avg,train_acc_avg,train_f1_avg,center_avg,lmcl_avg,cross_entropy_avg

    def evaluate(self, dataloader):
        self.model.eval()
        result = ""

        losses = 0
        total_correct = 0
        total_samples = 0
        f1_scores = 0

        center_avg = 0
        lmcl_avg = 0
        cross_entropy_avg = 0

        for x, y in tqdm(dataloader):
            x , y = x.to(config.DEVICE) , y.to(config.DEVICE)
            loss, accuracy, f1 , center_loss , lmcl_loss , cross_entropy = self._common_step((x, y), step='val')
            losses += loss.item()

            if self.use_center:
                center_avg += center_loss.item()
            else:
                center_avg = center_loss

            if self.use_lmcl:
                lmcl_avg += lmcl_loss.item()
            else:
                lmcl_avg = lmcl_loss

            if self.use_entropy:
                cross_entropy_avg += cross_entropy
            else:
                cross_entropy_avg = cross_entropy
                
            total_correct += accuracy * x.size(0)
            total_samples += x.size(0)
            f1_scores += f1

        try:
            loss_avg = losses / len(dataloader)
            result += f"Val Loss: {loss_avg:.4f} | "
        except:
            pass

        try:
            center_avg = center_avg / len(dataloader)
            result += f"Val Center Loss: {center_avg:.4f} | "
        except:
            # its not a number so we eliminate .4f
            result += f"Val Center Loss: {center_avg} | "
        
        try:
            lmcl_avg = lmcl_avg / len(dataloader)
            result += f"Val lmcl Loss: {lmcl_avg:.4f} | "
        except:
            result += f"Val lmcl Loss: {lmcl_avg} | "

        try:
            cross_entropy_avg = cross_entropy_avg / len(dataloader)
            acc_avg = total_correct / total_samples
            f1_avg = f1_scores / len(dataloader)
            result += f"Val Cross_entropy Loss: {cross_entropy_avg:.4f} | Val Acc: {acc_avg:.4f} | Val F1-Score: {f1_avg:.4f}"
        except:
            result += f"Val Cross_entropy Loss: {cross_entropy_avg}"

        print(result)
        return loss_avg, acc_avg, f1_avg,center_avg,lmcl_avg,cross_entropy_avg
    
    def set_value(self,var,val):
        self.data_dict[var] = val
        
    def fit(self, trainloader, val_loader, num_epochs=config.NUM_EPOCHS):

        val_pair_dataloader , test_pair_dataloader = create_pair_dataloader(
            val_pair_df=pd.read_csv('Dataset/val_pairs.csv'),
            test_pair_df=pd.read_csv('Dataset/test_pairs.csv')
        )
        
        for epoch in range(num_epochs):
            self.model.train()  # Set model to training mode
            train_loss, train_acc, train_f1 , train_center_loss , train_lmcl_loss , train_cross_loss = self.train(trainloader)
            # self.scheduler.step() # adjust learning rate

            self.model.eval()  # Set model to evaluation mode
            val_loss, val_acc, val_f1 , val_center_loss , val_lmcl_loss , val_cross_loss = self.evaluate(val_loader)

            roc_auc , f1 , precision , recall , accuracy , mean_cos_label_1 , mean_cos_label_0 = evaluate_metrics_knn(self.model,val_pair_dataloader)
            print(f"SimilarityDifference: {abs(mean_cos_label_1-mean_cos_label_0):.4f} | Mean Cosine Similarity for being Similar: {mean_cos_label_1:.4f} | Mean Cosine Similarity for not being similar: {mean_cos_label_0:.4f} | ROC-AUC-Score: {roc_auc:.4f} | Data_Pair_F1-Score: {f1:.4f} | Data_Pair_Precision: {precision:.4f} | Data_Pair_Recall: {recall:.4f} | Data_Pair_Accuracy: {accuracy:.4f}")

            self.set_value('train_loss',train_loss)
            self.set_value('train_acc',train_acc)
            self.set_value('train_f1',train_f1)
            self.set_value('train_center_loss',train_center_loss)
            self.set_value('train_lmcl_loss',train_lmcl_loss)
            self.set_value('train_cross_entropy_loss',train_cross_loss)
            self.set_value('val_loss',val_loss)
            self.set_value('val_acc',val_acc)
            self.set_value('val_f1',val_f1)
            self.set_value('val_center_loss',val_center_loss)
            self.set_value('val_lmcl_loss',val_lmcl_loss)
            self.set_value('val_cross_entropy_loss',val_cross_loss)
            self.set_value('val_pair_f1',f1)
            self.set_value('val_pair_precision',precision)
            self.set_value('val_pair_recall',recall)
            self.set_value('val_pair_accuracy',accuracy)
            self.set_value('val_pair_roc_auc',roc_auc)
            self.set_value('val_is_similar',mean_cos_label_1)
            self.set_value('val_not_similar',mean_cos_label_0)

            self.avg_meter.update(self.data_dict)
            self.data_dict = {}
            result = f"Epoch {epoch+1} | "

            try:
                result += f"Train_Loss: {train_loss:.4f} | "
            except:
                result += f"Train_Loss: {train_loss} | "

            try:
                result += f"Train_Center_Loss: {train_center_loss:.4f} | "
            except:
                result += f"Train_Center_Loss: {train_center_loss} | "

            try:
                result += f"Train_lmcl_loss: {train_lmcl_loss:.4f} | "
            except:
                result += f"Train_lmcl_loss: {train_lmcl_loss} | "

            try:
                result += f"Train CrossEntropyLoss: {train_cross_loss:.4f} | "
            except:
                result += f"Train CrossEntropyLoss: {train_cross_loss} | "

            result += f"Train_Accuracy: {train_acc:.4f} | Train_Fscore: {train_f1:.4f} | "


            try:
                result += f"Val_Loss: {val_loss:.4f} | "
            except:
                result += f"Val_Loss: {val_loss} | "

            try:
                result += f"Val_Center_Loss: {val_center_loss:.4f} | "
            except:
                result += f"Val_Center_Loss: {val_center_loss} | "

            try:
                result += f"Val_lmcl_loss: {val_lmcl_loss:.4f} | "
            except:
                result += f"Val_lmcl_loss: {val_lmcl_loss} | "

            try:
                result += f"Val CrossEntropyLoss: {val_cross_loss:.4f} | "
            except:
                result += f"Val CrossEntropyLoss: {val_cross_loss} | "

            result += f"Val_Accuracy: {val_acc:.4f} | Val_Fscore: {val_f1:.4f} | SimilarityDifference: {abs(mean_cos_label_1-mean_cos_label_0):.4f} | CosineSimilaritySimilar: {mean_cos_label_1:.4f} | CosineSimilarityNotSimilar: {mean_cos_label_0:.4f} | Similarity-ROC-AUC-Score: {roc_auc:.4f} | Similarity-F1-Score: {f1:.4f} | Similarity-Precision: {precision:.4f} | Similarity-Recall: {recall:.4f} | Similarity-Accuracy: {accuracy:.4f}\n"

            # result = f"Epoch {epoch+1} | Train_Loss: {train_loss:.4f} | Train_Center_Loss: {train_center_loss:.4f} | Train_lmcl_loss: {train_lmcl_loss:.4f} | Train CrossEntropyLoss: {train_cross_loss:.4f} | Train_Accuracy: {train_acc:.4f} | Train_Fscore: {train_f1:.4f} | Val_Loss: {val_loss:.4f} | Val_Center_Loss: {val_center_loss:.4f} | Val_lmcl_loss: {val_lmcl_loss:.4f} | Val CrossEntropyLoss: {val_cross_loss:.4f} | Val_Accuracy: {val_acc:.4f} | Val_Fscore: {val_f1:.4f} | SimilarityDifference: {abs(mean_cos_label_1-mean_cos_label_0):.4f} | CosineSimilaritySimilar: {mean_cos_label_1:.4f} | CosineSimilarityNotSimilar: {mean_cos_label_0:.4f} | Similarity-ROC-AUC-Score: {roc_auc:.4f} | Similarity-F1-Score: {f1:.4f} | Similarity-Precision: {precision:.4f} | Similarity-Recall: {recall:.4f} | Similarity-Accuracy: {accuracy:.4f}\n"
            log_training_process(result=result)

            # Save checkpoint after each epoch
            checkpoint_path = config.CHECKPOINT_FILE
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                epoch=epoch+1,
                train_loss=train_loss,
                valid_loss=val_loss,
                train_acc=train_acc,
                valid_acc=val_acc,
                center_loss_checkpoint=self.center_loss,
                valid_f1=val_f1,
                cosine_similarity_true=mean_cos_label_1,
                cosine_similarity_false=mean_cos_label_0,
                f1_score_similarity=f1,
                roc_auc_similarity=roc_auc,
                precision_similarity=precision,
                recall_similarity=recall
            )

        print("Training finished.")

        test_roc_auc, test_f1, test_precision, test_recall, test_accuracy, test_mean_cos_sim_label_1, test_mean_cos_sim_label_0 = evaluate_metrics_knn(self.model, test_pair_dataloader)

        print("Testset: ROC AUC:", test_roc_auc)
        print("Testset: F1 Score:", test_f1)
        print("Testset: Precision:", test_precision)
        print("Testset: Recall:", test_recall)
        print("Testset: Accuracy:", test_accuracy)
        print("Testset: Mean Cosine Similarity for label 1:", test_mean_cos_sim_label_1)
        print("Testset: Mean Cosine Similarity for label 0:", test_mean_cos_sim_label_0)

        return self.avg_meter
    

# import torchvision
# import torch.nn.functional as F
# model = torchvision.models.vgg16(pretrained=True)
# modules = nn.Sequential(*list(model.children())[:-2])
# x = modules(torch.randn(1,3,150,150))
# x = F.adaptive_avg_pool2d(x, (1, 1))
# x = torch.flatten(x, 1)
>>>>>>> 03dc744fd576ee5506938c3e230e3f73e0db4d51
>>>>>>> ca7d9da84a74cc9f1a85f0eb812d853e11a9db46
