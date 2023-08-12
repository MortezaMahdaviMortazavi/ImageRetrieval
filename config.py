<<<<<<< HEAD
import torch
import argparse

parser = argparse.ArgumentParser(description='Script Description')

# # File Paths
# parser.add_argument('--images_folder', type=str, default='Dataset/jpg', help='Path to the folder containing images')
# parser.add_argument('--annotation_file', type=str, default='Dataset/imagelabels.mat', help='Path to the annotation file')
# parser.add_argument('--txt_results', type=str, default='logs/logs.txt', help='Path to the text results file')
# parser.add_argument('--logfile', type=str, default='logs/model.pt', help='Path to the log file')
# parser.add_argument('--csv_dataset_file', type=str, default='Dataset/dataset.csv', help='Path to the CSV dataset file')
# parser.add_argument('--cat_to_name_file', type=str, default='Dataset/cat_to_name.json', help='Path to the JSON file mapping category to name')
# parser.add_argument('--checkpoint_file', type=str, default='logs/checkpoints.pt', help='Path to the checkpoint file')
# parser.add_argument('--train_file', type=str, default='Dataset/train_dataset.csv', help='Path to the train dataset file')
# parser.add_argument('--valid_pairs_file', type=str, default='Dataset/val_augmented_pair.csv', help='Path to the validation pairs file')
# parser.add_argument('--test_pairs_file', type=str, default='Dataset/test_augmented_pair.csv', help='Path to the test pairs file')
# parser.add_argument('--data_folder', type=str, default='Dataset/jpg', help='Path to the folder containing data')

# # Device and Random State
# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
# parser.add_argument('--random_state', type=int, default=47, help='Random seed for reproducibility')

# # Test Size and Image Size
# parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset to include in the test split')
# parser.add_argument('--image_size', type=int, nargs=2, default=[150, 150], help='Image size (width, height) for resizing')

# # Number of Retrieves
# parser.add_argument('--number_of_retrieves', type=int, default=50, help='Number of images to retrieve in retrieval task')

# # Hyperparameters
# parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
# parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs for training')

# # Center Loss Hyperparameters
# parser.add_argument('--center_learning_rate', type=float, default=0.3, help='Learning rate for center loss')
# parser.add_argument('--num_classes', type=int, default=82, help='Number of classes in the dataset')
# parser.add_argument('--alpha', type=float, default=1, help='Weight for center loss')
# parser.add_argument('--feature_dim', type=int, default=2048, help='Feature dimension for center loss')
# parser.add_argument('--lambda_', type=int, default=10, help='Weight for center loss regularization')

# args = parser.parse_args()

IMAGES_FOLDER = 'Dataset/jpg'
ANNOTATION_FILE = 'Dataset/imagelabels.mat'
TXT_RESULTS = 'logs/logs.txt'
LOGFILE = 'logs/model.pt'
CSV_DATASET_FILE = 'Dataset/dataset.csv'
CAT_TO_NAME_FILE = 'Dataset/cat_to_name.json'
CHECKPOINT_FILE = 'logs/checkpoints.pt'
TRAIN_FILE = 'Dataset/train_dataset.csv'
VALID_PAIRS_FILE = 'Dataset/val_augmented_pair.csv'
TEST_PAIRS_FILE = 'Dataset/test_augmented_pair.csv'
DATA_FOLDER = 'Dataset/jpg'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cuda'
# DEVICE = 'cpu'
RANDOM_STATE = 47
TEST_SIZE = 0.2
IMAGE_SIZE = (128,128)
NUMBER_OF_RETRIEVES = 50
# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.005
NUM_EPOCHS= 100


# Losses Hyperparameters
CENTER_LEARNING_RATE = 0.3
NUM_CLASSES = 82
ALPHA = 1
FEATURE_DIM = 8192

LAMBDA_C = 0.01
MARGIN = 0.5
SCALE = 30.0


# models config
EFFICIENT_NETV2_IN_FEATURES = 1280

=======
import torch
import argparse

parser = argparse.ArgumentParser(description='Script Description')

# # File Paths
# parser.add_argument('--images_folder', type=str, default='Dataset/jpg', help='Path to the folder containing images')
# parser.add_argument('--annotation_file', type=str, default='Dataset/imagelabels.mat', help='Path to the annotation file')
# parser.add_argument('--txt_results', type=str, default='logs/logs.txt', help='Path to the text results file')
# parser.add_argument('--logfile', type=str, default='logs/model.pt', help='Path to the log file')
# parser.add_argument('--csv_dataset_file', type=str, default='Dataset/dataset.csv', help='Path to the CSV dataset file')
# parser.add_argument('--cat_to_name_file', type=str, default='Dataset/cat_to_name.json', help='Path to the JSON file mapping category to name')
# parser.add_argument('--checkpoint_file', type=str, default='logs/checkpoints.pt', help='Path to the checkpoint file')
# parser.add_argument('--train_file', type=str, default='Dataset/train_dataset.csv', help='Path to the train dataset file')
# parser.add_argument('--valid_pairs_file', type=str, default='Dataset/val_augmented_pair.csv', help='Path to the validation pairs file')
# parser.add_argument('--test_pairs_file', type=str, default='Dataset/test_augmented_pair.csv', help='Path to the test pairs file')
# parser.add_argument('--data_folder', type=str, default='Dataset/jpg', help='Path to the folder containing data')

# # Device and Random State
# parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
# parser.add_argument('--random_state', type=int, default=47, help='Random seed for reproducibility')

# # Test Size and Image Size
# parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of dataset to include in the test split')
# parser.add_argument('--image_size', type=int, nargs=2, default=[150, 150], help='Image size (width, height) for resizing')

# # Number of Retrieves
# parser.add_argument('--number_of_retrieves', type=int, default=50, help='Number of images to retrieve in retrieval task')

# # Hyperparameters
# parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
# parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
# parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs for training')

# # Center Loss Hyperparameters
# parser.add_argument('--center_learning_rate', type=float, default=0.3, help='Learning rate for center loss')
# parser.add_argument('--num_classes', type=int, default=82, help='Number of classes in the dataset')
# parser.add_argument('--alpha', type=float, default=1, help='Weight for center loss')
# parser.add_argument('--feature_dim', type=int, default=2048, help='Feature dimension for center loss')
# parser.add_argument('--lambda_', type=int, default=10, help='Weight for center loss regularization')

# args = parser.parse_args()

IMAGES_FOLDER = 'Dataset/jpg'
ANNOTATION_FILE = 'Dataset/imagelabels.mat'
TXT_RESULTS = 'logs/logs.txt'
LOGFILE = 'logs/model.pt'
CSV_DATASET_FILE = 'Dataset/dataset.csv'
CAT_TO_NAME_FILE = 'Dataset/cat_to_name.json'
CHECKPOINT_FILE = 'logs/checkpoints.pt'
TRAIN_FILE = 'Dataset/train_dataset.csv'
VALID_PAIRS_FILE = 'Dataset/val_augmented_pair.csv'
TEST_PAIRS_FILE = 'Dataset/test_augmented_pair.csv'
DATA_FOLDER = 'Dataset/jpg'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cuda'
# DEVICE = 'cpu'
RANDOM_STATE = 47
TEST_SIZE = 0.2
IMAGE_SIZE = (128,128)
NUMBER_OF_RETRIEVES = 50
# Hyperparameters
BATCH_SIZE = 512
LEARNING_RATE = 0.005
NUM_EPOCHS= 100


# Losses Hyperparameters
CENTER_LEARNING_RATE = 0.3
NUM_CLASSES = 82
ALPHA = 1
FEATURE_DIM = 512

LAMBDA_C = 0.01
MARGIN = 0.5
SCALE = 30.0


# models config
EFFICIENT_NETV2_IN_FEATURES = 1280

>>>>>>> ca7d9da84a74cc9f1a85f0eb812d853e11a9db46
