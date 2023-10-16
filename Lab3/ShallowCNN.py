import os
import time

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from Lab1.data_explorer import splitDataset

from typing import List

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


class ShallowCNN(nn.Module):
    def __init__(
            self,
            num_classes: List[int],
            device: str = 'cpu'
    ):
        super(ShallowCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 2**3, kernel_size = 5, stride = 1, padding = 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 2**3, out_channels = 2**6, kernel_size = 5, stride = 1, padding = 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(2**6 * 8 * 8, num_classes)
        )
        self.to(device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
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

def eval(model: nn.Module, testDataLoader, device='cpu'):
    model.eval()  
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


def plot_and_save_loss(train_curve, vali_curve, save_path='./plots/', title="Training & Validation Loss Curves"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_curve, label='Training Loss', color='blue')
    plt.plot(vali_curve, label='Validation Loss', color='red')
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves")
    
    plt.legend()
    
    plt.savefig(os.path.join(save_path, f'{title}.png'))


if __name__ == "__main__":
    PATH_TO_DATASETS = './dataset/'
    if not os.path.exists(PATH_TO_DATASETS):
        os.makedirs(PATH_TO_DATASETS)
    
    PATH_TO_PLOTS = "./Lab3/plots/"
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
    
    # Hyperparameter combinations
    BATCH_SIZES = [32, 64]
    EPOCHS = [100, 200]
    LEARNING_RATES = [0.001, 0.0001]
    
    all_train_curves = {}
    all_val_curves = {}


    transform = transforms.Compose([transforms.ToTensor(), 
                        transforms.Lambda(lambda x: x.to(device)),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    dataset = torchvision.datasets.CIFAR10(root=PATH_TO_DATASETS, download=True, transform=transform)
    
    for BATCH_SIZE in BATCH_SIZES:
        for EPOCH in EPOCHS:
            for LEARNING_RATE in LEARNING_RATES:
                train_dataloader, val_dataloader, test_dataloader = getTrainValTestData(dataset=dataset, 
                                                                                        batch_size=BATCH_SIZE,
                                                                                        val_ratio=VALIDATION_RATIO, 
                                                                                        test_ratio=TEST_RATIO, 
                                                                                        subset_ratio=SUBSET_RATIO,)
                num_classes = len(dataset.classes)
                
                model = ShallowCNN(num_classes=num_classes, device=device)
                print(model)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

                train_curve, vali_curve= train(model, train_dataloader, val_dataloader,criterion, optimizer, EPOCH, RUNNING_BATA, device)
                
                Pltitle = f"B{BATCH_SIZE}-E{EPOCH}-L{LEARNING_RATE}-I"
                all_train_curves[Pltitle] = train_curve
                all_val_curves[Pltitle] = vali_curve
                
                test_accuracy = eval(model, test_dataloader, device)
                print(f'Accuracy of the model on the test images: {test_accuracy:.3f}%')
                with open('./Lab3/test_accuracy.txt', 'a') as file:
                    file.write(f'Test Accuracy of {Pltitle}: {test_accuracy:.3f}%\n')

    plt.figure(figsize=(10, 5))
    for title, curve in all_train_curves.items():
        plt.plot(curve, label=f'{title} Train')
    for title, curve in all_val_curves.items():
        plt.plot(curve, label=f'{title} Val', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curves for All Hyperparameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_TO_PLOTS, 'all_curves.png'))