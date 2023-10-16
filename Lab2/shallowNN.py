import os
import time

import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from Lab1.data_explorer import splitDataset

from typing import List

class ShallowNN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: int,
            num_classes: List[int],
            dropout_prob: float,
            device: str = 'cpu'
    ):
        super(ShallowNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = dim

        self.hidden_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_dim, num_classes)
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        #x = x.to(device=self.device)
        x = torch.flatten(x, 1)
        x = self.hidden_layers(x)
        x = self.fc(x)
        return x

def train(model: nn.Module, 
          trainloader,
          valoader,
          criterion,
          optimizer,
          epoches: int,
          beta:int = 0.1,
          device = 'cpu',
    ):
    train_curve = []
    vali_curve = []
    best_val_loss = float('inf')
    best_model = None
    start_time = time.time()

    for epoch in range(epoches):
        running_loss = 0.0
        #with torch.device(device):
        for data in trainloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = criterion(out, labels)
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()
            running_loss = beta*loss.item() + (1-beta)*running_loss
        
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for data in valoader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                total_val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
                
        train_curve.append(running_loss)
        vali_curve.append(avg_val_loss)
        print('Epoch [{}/{}], Loss: {:.4f},  Validation Loss: {:.4f}'.format(epoch+1, epoches, running_loss, avg_val_loss)) 

    end_time = time.time()
    model.load_state_dict(best_model)
    print(f"Traing finished! cost {end_time-start_time:.2f} seconds")
    return train_curve, vali_curve
                                                                             
def getTrainValTestData(dataset, batch_size, val_ratio, test_ratio, subset_ratio):
    train_indices, val_indices, test_indices, _, _, _= splitDataset(np.arange(len(dataset)), np.arange(len(dataset)), val_ratio, test_ratio, 42)
    
    train_subset = torch.utils.data.Subset(dataset, train_indices[:getSubsize(subset_ratio, train_indices)])
    val_subset = torch.utils.data.Subset(dataset, val_indices[:getSubsize(subset_ratio, val_indices)])
    test_subset = torch.utils.data.Subset(dataset, test_indices[:getSubsize(subset_ratio, test_indices)])
    
    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def getSubsize(subsetRatio, indices):
    return int(len(indices)*subsetRatio)

def plot_and_save_loss(train_curve, vali_curve, save_path='./plots/'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_curve, label='Training Loss', color='blue')
    plt.plot(vali_curve, label='Validation Loss', color='red')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    
    plt.legend()
    
    plt.savefig(os.path.join(save_path, f'Trainging_validation curve.png'))


def eval(model: nn.Module, testDataLoader, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testDataLoader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    PATH_TO_DATASETS = './dataset/'
    if not os.path.exists(PATH_TO_DATASETS):
        os.makedirs(PATH_TO_DATASETS)
    
    PATH_TO_PLOTS = "./Lab2/plots/"
    if not os.path.exists(PATH_TO_PLOTS):
        os.makedirs(PATH_TO_PLOTS)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    #device = 'cpu'
    print(f"Training on {device}.")

    SUBSET_RATIO = 0.1
    VALIDATION_RATIO = 0.2
    TEST_RATIO = 0.1
    RUNNING_BATA = 0.05
    
    BATCH_SIZE = 64
    EPOCH = 100
    LEARNING_RATE = 0.001
    HIDDEN_LAYERS: List[int] = [512, 256, 128]
    DROPOUT_RATIO = 0.5
    WEIGHT_DECAY = 0.0001

    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Lambda(lambda x: x.to(device)),
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    dataset = torchvision.datasets.CIFAR10(root=PATH_TO_DATASETS, download=True, transform=transform)
    train_dataloader, val_dataloader, test_dataloader = getTrainValTestData(dataset=dataset, 
                                                                            batch_size=BATCH_SIZE,
                                                                            val_ratio=VALIDATION_RATIO, 
                                                                            test_ratio=TEST_RATIO, 
                                                                            subset_ratio=SUBSET_RATIO,)

    input_size = dataset[0][0].view(-1).size(0)
    num_classes = len(dataset.classes)
    
    model = ShallowNN(input_dim=input_size, hidden_dims=HIDDEN_LAYERS, num_classes=num_classes, dropout_prob=DROPOUT_RATIO, device=device)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_curve, vali_curve= train(model, train_dataloader, val_dataloader,criterion, optimizer, EPOCH, RUNNING_BATA, device)
    plot_and_save_loss(train_curve=train_curve, vali_curve=vali_curve, save_path=PATH_TO_PLOTS)
    print(f'Accuracy of the model on the test images: {eval(model, test_dataloader, device):.2f}%')