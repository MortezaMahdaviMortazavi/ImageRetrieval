import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')

class AverageMeter(object):
    def __init__(self):
        self.metrics = {}
        """
            Metrics:
                train_loss, 
                train_acc, 
                train_f1 , 
                train_center_loss , 
                train_cross_loss ,
                val_loss, val_acc, 
                val_f1 , 
                val_center_loss , 
                val_cross_loss
                val_pair_roc_auc , 
                val_pair_f1 , 
                val_pair_precision , 
                val_pair_recall , 
                val_pair_accuracy , 
                val_pair_is_similar , 
                val_pair_not_similar 
        
        """

    def reset(self):
        self.metrics = {}

    def update(self, data_dict):
        for metric_name, value in data_dict.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = [value]
            else:
                self.metrics[metric_name].append(value)

    def get_metric(self, metric_name):
        return np.array(self.metrics[metric_name])

    def get_average_metric(self, metric_name):
        return np.mean(self.metrics[metric_name])

    def plot_metric(self, metric_name):
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' does not exist.")

        metric_values = self.get_metric(metric_name)
        epochs = range(1, len(metric_values) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, metric_values, marker='o', color='b')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.title(f"{metric_name} over Iterations", fontsize=16)
        plt.grid(True)
        plt.show()


    def plot_all_metrics(self):
        num_metrics = len(self.metrics)
        num_rows = int(num_metrics / 2) + num_metrics % 2
        num_cols = 2

        plt.figure(figsize=(15, 5 * num_rows))

        for i, metric_name in enumerate(self.metrics):
            metric_values = self.get_metric(metric_name)
            epochs = range(1, len(metric_values) + 1)

            plt.subplot(num_rows, num_cols, i + 1)
            plt.plot(epochs, metric_values, marker='o', color='b')
            plt.xlabel('Epochs', fontsize=14)
            plt.ylabel(metric_name, fontsize=14)
            plt.title(f"{metric_name} over Iterations", fontsize=16)
            plt.grid(True)

        plt.tight_layout()
        plt.show()
