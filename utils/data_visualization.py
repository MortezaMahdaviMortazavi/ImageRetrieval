import matplotlib.pyplot as plt
import random
from PIL import Image
import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')
import config

def plot_images_with_labels(data_df, num_images=5, random_seed=42, show_image_mode=True):
    """
    Plot some images with their corresponding labels from the given DataFrame.

    Args:
        data_df (pd.DataFrame): The DataFrame containing 'filepaths' and 'labels' columns.
        num_images (int, optional): Number of images to plot. Defaults to 5.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        show_image_mode (bool, optional): Whether to show image mode (RGB or not) alongside the labels. 
                                          Defaults to True.
    """
    random.seed(random_seed)
    sampled_data = data_df.sample(n=num_images)

    num_rows = (num_images - 1) // 5 + 1
    num_cols = min(num_images, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    for idx, (filepath, label) in enumerate(zip(sampled_data['image_file'], sampled_data['label'])):
        image = Image.open(filepath)

        row_idx = idx // 5
        col_idx = idx % 5

        axes[row_idx, col_idx].imshow(image)
        axes[row_idx, col_idx].axis('off')

        if show_image_mode:
            is_rgb = image.mode == 'RGB'
            axes[row_idx, col_idx].set_title(f'Label: {label} | RGB: {is_rgb}')
        else:
            axes[row_idx, col_idx].set_title(f'Label: {label}')

    for idx in range(num_images, num_rows * 5):
        row_idx = idx // 5
        col_idx = idx % 5
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.show()