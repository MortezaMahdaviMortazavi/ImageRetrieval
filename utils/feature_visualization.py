from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch

from tqdm import tqdm

import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')
import config


def extract_features(model,dataloader):
    model.eval()
    all_features = []

    
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.cuda().unsqueeze(0)
            features, _ = model(images)
            all_features.append(features.cpu())
    
    all_features = torch.cat(all_features, dim=0)
    torch.save(all_features, 'checkpoints/BigBatch_test_gallery.pt')
    return all_features

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane using PCA for dimensionality reduction.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    # Apply PCA to reduce the features to 2D
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # Generate a list of colors for each class
    colors = plt.cm.get_cmap('tab20', num_classes)

    # Plot each class separately with a different color
    for label_idx in range(num_classes):
        plt.scatter(
            reduced_features[labels==label_idx, 0],
            reduced_features[labels==label_idx, 1],
            c=[colors(label_idx)],
            s=1,
            label=str(label_idx)
        )
    
    plt.legend(loc='upper right')
    dirname = os.path.join(config, prefix)
    if not os.exists(dirname):
        os.mkdir(dirname)
    save_name = os.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


from sklearn.cluster import KMeans

def TSNe_plot(data):
    labels = np.array(pd.read_csv('Dataset/test_dataset.csv')['label'].tolist())
    perplexity_value = 30

    tsne_model = TSNE(n_components=2, random_state=42,perplexity=perplexity_value)
    transformed_data = tsne_model.fit_transform(data)
    tsne_df = pd.DataFrame(data=transformed_data, columns=['tsne_component_1', 'tsne_component_2'])
    tsne_df['label'] = labels

    n_clusters = 20  # Number of clusters, you can adjust this based on your classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(transformed_data)

    # Save the DataFrame to a CSV file
    custom_cmap = plt.cm.get_cmap('tab20', len(np.unique(labels)))

    tsne_df.to_csv('transformed_data.csv', index=False)
    plt.figure(figsize=(30, 30))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap=custom_cmap)
    plt.colorbar()
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    for cluster in range(n_clusters):
        cluster_center = kmeans.cluster_centers_[cluster]
        cluster_members = transformed_data[cluster_labels == cluster]
        for member in cluster_members:
            plt.plot([cluster_center[0], member[0]], [cluster_center[1], member[1]], 'k-', alpha=0.2)

    plt.show()
