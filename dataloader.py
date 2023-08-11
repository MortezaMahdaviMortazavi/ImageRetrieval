import torch
import config
import torchvision.transforms as transforms
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from sklearn.model_selection import train_test_split

class FlowerDataset(torch.utils.data.Dataset):
    def __init__(self,df,transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        filepath = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]

        img = Image.open(filepath)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label)
        return img , label
    

class FlowerPairDataset(torch.utils.data.Dataset):
    def __init__(self,df,transform):
        self.df = df
        self.transform = transform

        self.img1_lists = self.df['image1'].tolist()
        self.img2_lists = self.df['image2'].tolist()
        self.labels = self.df['similar'].tolist()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        img1 = Image.open(self.img1_lists[idx])
        img2 = Image.open(self.img2_lists[idx])
        similarity_label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1 , img2 , torch.tensor(similarity_label)


class FlowerDataModule(pl.LightningDataModule):
    def __init__(self, train_df=None, test_df=None, data_dir='jpg/', batch_size=32):
        super(FlowerDataModule, self).__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = None

    def prepare_data(self):
        # download data
        # tokenize
        # etc
        pass

    def _setup_transform(self):
        self.transform = {
            'train' : transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(30),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
                                                                    
            'val' : transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
        }

    def setup(self, stage=None):
        transform = self.transform

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if self.train_df is None and self.test_df is None:
            dataset = pd.read_csv(config.CSV_DATASET_FILE)
            self.train_df, self.test_df = train_test_split(dataset, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=dataset['label'])

        elif self.train_df is None and self.test_df is not None:
            self.train_df, self.test_df = train_test_split(self.test_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=self.test_df['label'])

        trainset = FlowerDataset(self.train_df, transform)
        testset = FlowerDataset(self.test_df, transform)

        self.trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.val_dataloader
    
    @property
    def get_train_dataloader(self):
        return self.trainloader
    
    @property
    def get_val_dataloader(self):
        return self.val_dataloader
    

def create_normal_dataloader(train_df,test_df):
    val_transforms = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Additional augmentations for training data
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if train_df is None and test_df is None:
        # dataset = pd.read_csv(config.CSV_DATASET_FILE)
        dataset = pd.read_csv(config.TRAIN_FILE)
        train_df , test_df = train_test_split(dataset,test_size=config.TEST_SIZE,random_state=config.RANDOM_STATE,stratify=dataset['label'])
        print("num_classes of dataset",len(list(set(dataset['label'].tolist()))))
        
    elif train_df is None and test_df is not None:
        train_df , test_df = train_test_split(test_df,test_size=config.TEST_SIZE,random_state=config.RANDOM_STATE,stratify=test_df['label'])

  
    elif train_df is not None and test_df is None:
        train_df , test_df = train_test_split(train_df,test_size=config.TEST_SIZE,random_state=config.RANDOM_STATE,stratify=train_df['label'])
      


    trainset = FlowerDataset(train_df, train_transforms)
    testset = FlowerDataset(test_df, val_transforms)

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    return trainloader, testloader



def create_pair_dataloader(train_pair_df=None,val_pair_df=None,test_pair_df=None):
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    if val_pair_df is None:
        val_pair_df = pd.read_csv(config.VALID_PAIRS_FILE)

    if test_pair_df is None:
        test_pair_df = pd.read_csv(config.TEST_PAIRS_FILE)

    if train_pair_df is None:
        train_pair_df = pd.read_csv('Dataset/train_pairs.csv')

    train_pair_dataset = FlowerPairDataset(train_pair_df,transform)
    val_pair_dataset = FlowerPairDataset(val_pair_df,transform)
    test_pair_dataset = FlowerPairDataset(test_pair_df,transform)

    train_pair_dataloader = torch.utils.data.DataLoader(train_pair_dataset,batch_size=config.BATCH_SIZE,shuffle=False)
    val_pair_dataloader = torch.utils.data.DataLoader(val_pair_dataset,batch_size=config.BATCH_SIZE,shuffle=False)
    test_pair_dataloader = torch.utils.data.DataLoader(test_pair_dataset,batch_size=config.BATCH_SIZE,shuffle=False)

    return train_pair_dataloader,val_pair_dataloader , test_pair_dataloader
