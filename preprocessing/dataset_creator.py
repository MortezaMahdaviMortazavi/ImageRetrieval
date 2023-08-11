import sys
sys.path.append('C:/Users/pc\projects/AiRoshanIntenrship/SimilarityProject')
import config
import scipy
import os
from sklearn.model_selection import train_test_split
import pandas as pd


def create_dataset(img_folder=None, annotation_file=None):
    if img_folder is None:
        img_folder = config.IMAGES_FOLDER
    
    if annotation_file is None:
        annotation_file = config.ANNOTATION_FILE

    data = scipy.io.loadmat(annotation_file)
    labels = data['labels'].tolist()

    image_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]

    img_paths = []
    img_labels = []

    index = 0
    for image_file in image_files:
        img_path = os.path.join(img_folder, image_file)
        img_label = labels[0][index]
        index+=1

        img_paths.append(img_path)
        img_labels.append(img_label)

    # Create the DataFrame
    df = pd.DataFrame({'ImagePaths': img_paths, 'Labels': img_labels})
    train_df, test_df = train_test_split(df, stratify=df['Labels'], test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    return train_df , test_df