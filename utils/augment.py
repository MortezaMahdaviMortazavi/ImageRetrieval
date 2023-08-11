import cv2
import random
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')



def random_resize(image, min_scale=0.8, max_scale=1.2):
    scale_factor = random.uniform(min_scale, max_scale)
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height))

def random_flip(image, p=0.5):
    if random.random() < p:
        return cv2.flip(image, 1)  # Horizontal flip
    return image

def random_rotation(image, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def random_brightness(image, max_delta=30):
    delta = random.randint(-max_delta, max_delta)
    return cv2.add(image, delta)

def random_contrast(image, min_factor=0.8, max_factor=1.2):
    factor = random.uniform(min_factor, max_factor)
    return cv2.multiply(image, factor)

def sequential_augmentation(image):
    image = random_resize(image)
    image = random_flip(image)
    image = random_rotation(image)
    image = random_brightness(image)
    # image = random_contrast(image)
    return image

def opencv_augmentation(image):
    
    image_size = 224
    degrees = 30
    brightness = 0.4
    contrast = 0.4
    saturation = 0.4
    hue = 0.2
    # Random rotation
    angle = np.random.uniform(-degrees, degrees)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Color jitter
    hsv = cv2.cvtColor(rotated, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * np.random.uniform(1 - saturation, 1 + saturation)
    hsv[:, :, 2] = hsv[:, :, 2] * np.random.uniform(1 - brightness, 1 + brightness)
    hsv[:, :, 0] = hsv[:, :, 0] + np.random.uniform(-hue, hue) * 180
    jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Convert to tensor
    augmented_img = jittered.astype(np.float32) #/ 255.0
    return augmented_img




def augment_class(df, class_label, max_samples, save_path):
    class_df = df[df['label'] == class_label]
    
    class_save_path = os.path.join(save_path, str(class_label))
    if not os.path.exists(class_save_path):
        os.makedirs(class_save_path)
    
    num_samples = max_samples - len(class_df)
    
    augmented_data = []
    
    for i in range(num_samples):
        row = class_df.sample(n=1).iloc[0]

        image_path = row['image_file']
        image = cv2.imread(image_path)
        image = opencv_augmentation(image)

        save_filename = os.path.join(class_save_path, f"{class_label}_{i+1}.jpg")
        cv2.imwrite(save_filename, image)
        
        augmented_data.append({'image_file': save_filename, 'label': class_label})

    augmented_df = pd.DataFrame(augmented_data)

    print(f"Augmentation for class {class_label} completed.")
    
    return augmented_df


def augment_all_classes(df, max_samples, save_path):
    augmented_dfs = []
    unique_labels = df['label'].unique()

    for class_label in tqdm(unique_labels):
        augmented_df = augment_class(df, class_label, max_samples, save_path)

        augmented_dfs.append(augmented_df)

    augmented_df = pd.concat(augmented_dfs, ignore_index=True)

    csv_path = os.path.join(save_path, 'augmented_data.csv')
    augmented_df.to_csv(csv_path, index=False)

    return augmented_df