import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')

import scipy
import config
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def data_splitter(train_dataset,test_size=None):
    if test_size is None:
        test_size = config.TEST_SIZE

    largeset , smallset = train_test_split(
        train_dataset,test_size=test_size,
        random_state=config.RANDOM_STATE,
        stratify=train_dataset['label'])
    
    return largeset , smallset


def preprocess():
    img_labels = scipy.io.loadmat(config.ANNOTATION_FILE)
    img_labels = img_labels["labels"]
    img_labels = img_labels[0]
    for i in range(len(img_labels)):
        img_labels[i] = img_labels[i] - 1

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    # file = config.CSV_DATASET_FILE
    # dataset = pd.read_csv(file)
    dir = config.IMAGES_FOLDER
    unique_labels = np.unique(img_labels) # also sort the classes base on their numbers

    # Randomly select 20 classes for testi
    # val_classes = np.random.choice(unique_labels, size=20, replace=False)
    val_classes = unique_labels[-20:]
    unique_labels = unique_labels[:-20]

    for imgs in tqdm(os.listdir(dir)):
        img_num = int(imgs[7:11]) - 1
        if img_labels[img_num] in val_classes:
            test_y.append(img_labels[img_num])
            image = os.path.join(dir, imgs)
            test_x.append(image)
        else:
            train_y.append(img_labels[img_num])
            image = os.path.join(dir, imgs)
            train_x.append(image)

    # Training data
    train_data = {'image_file': train_x, 'label': train_y}
    train_df = pd.DataFrame(train_data)
    # testidation data
    test_data = {'image_file': test_x, 'label': test_y}
    test_df = pd.DataFrame(test_data)
    test_df , val_df = data_splitter(train_dataset=test_df,test_size=0.2)

    train_output_csv_filename = 'Dataset/train_dataset.csv'
    val_output_csv_filename = 'Dataset/val_dataset.csv'
    test_output_csv_filename = 'Dataset/test_dataset.csv'

    train_df.to_csv(train_output_csv_filename, index=False)
    val_df.to_csv(val_output_csv_filename,index=False)
    test_df.to_csv(test_output_csv_filename, index=False)

    print("Training dataset (First 5 rows):")
    print(train_df.head())
    print("\nValidation dataset (First 5 rows):")
    print(val_df.head())
    print("\nTest dataset (First 5 rows):")
    print(test_df.head())