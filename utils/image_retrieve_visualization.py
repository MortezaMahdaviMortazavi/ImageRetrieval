import torch
import torchvision.transforms as transforms
import torch.nn as nn
import PIL
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')
import config

class FlowerDataset:
    pass


def retrieve_similar_images(image_path: str, model, df):
    """
    Given an input image, a trained Discriminator model, and a folder containing images,
    return the 10 most similar images to the input image using cosine similarity.

    :param image_path: The path to the input image
    :param model: A trained Discriminator model
    :param data_folder: The path to the folder containing images
    :return: A list of tuples containing the 10 most similar images and their cosine similarity scores
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Load the input image
    input_image = transform(PIL.Image.open(image_path)).cuda()

    # Extract features from the input image
    model.eval()
    with torch.no_grad():
        image_features, _ = model(input_image.unsqueeze(0))

    cosine_similarity = nn.CosineSimilarity(dim=1)

    similarities = []
    # Load and process images from the data folder
    dataset = FlowerDataset(df,transform=transform)
    for sample in tqdm(dataset):
        sample_features, _ = model(sample[0].unsqueeze(0).cuda())
        similarity = cosine_similarity(sample_features, image_features)
        similarities.append((sample[0].permute(1, 2, 0), similarity.item()))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:config.NUMBER_OF_RETRIEVES]

def plot_similar_images(model,img_path,rows=5, cols=10, figsize=(30, 30)):
    """
    Plot a list of image tensors in a grid layout.

    :param similar_images: A list of image tensors to be plotted.
    :param rows: Number of rows in the grid layout.
    :param cols: Number of columns in the grid layout.
    :param figsize: Size of the figure (width, height) in inches.
    """
    similar_images = retrieve_similar_images(img_path,model,df=pd.read_csv(config.CSV_DATASET_FILE))
    similar_images = list(map(lambda t: t[0], similar_images))

    num_images = len(similar_images)
    for idx in range(num_images):
        similar_images[idx] = reverse_normalization(similar_images[idx].permute(2,0,1)).permute(1,2,0)
    total_plots = rows * cols

    if total_plots < num_images:
        print(f"Warning: Number of images exceeds the grid size. Only plotting first {total_plots} images.")

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.4)

    for i in range(total_plots):
        if i < num_images:
            img_tensor = similar_images[i].numpy()
            # Convert axes to a 2-dimensional array if there is only one row
            if rows == 1:
                ax = axes[i % cols]  # Access the correct subplot
            else:
                ax = axes[i // cols, i % cols]  # Access the correct subplot

            ax.imshow(img_tensor)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.show()

def reverse_normalization(normalized_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Reverse the normalization of an image.

    :param normalized_image: A tensor representing the normalized image.
    :param mean: List of mean values used in the normalization.
    :param std: List of standard deviation values used in the normalization.
    :return: The original image.
    """
    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    original_image = normalized_image * std + mean
    return original_image

