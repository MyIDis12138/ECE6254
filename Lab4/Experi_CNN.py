# Experiments for CNN model
import os

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}.")

PATH_TO_PLOTS = "./Lab4/plots/"
if not os.path.exists(PATH_TO_PLOTS):
    os.makedirs(PATH_TO_PLOTS)

def set_seed(seed_value=44):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloader(train_dataset_root, val_dataset_root, test_dataset_root, batch_size, transform):
    train_dataset = datasets.ImageFolder(root=train_dataset_root, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dataset_root, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dataset_root, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class BaseCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 2
    ):
        super(BaseCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 2**3, kernel_size = 5, stride = 1, padding = 2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(2**3 * 112 * 112, num_classes)
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x

class DropoutCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
            dropout_rate: float = 0.5
    ):
        super(DropoutCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 2**3, kernel_size = 5, stride = 1, padding = 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(2**3 * 112 * 112, num_classes)
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x

class BatchNormCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
    ):
        super(BatchNormCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 2**3, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(2**3),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(2**3 * 112 * 112, num_classes)
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x

class BestCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.5
    ):
        super(BestCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 2**3, kernel_size = 5, stride = 1, padding = 2),
            nn.BatchNorm2d(2**3),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(2**3 * 112 * 112, num_classes)
        )
        self.to(device)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = F.softmax(x, dim=1)
        return x

def train(model, train_loader, val_loader, epochs, optimizer, criterion):
    train_loss = []
    val_loss = []
    bestModel = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss.append(running_loss/len(train_loader))

        model.eval()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
        
        avg_val_loss = running_loss/len(val_loader)
        if avg_val_loss < min(val_loss, default=float('inf')):
            bestModel = model.state_dict()
        val_loss.append(avg_val_loss)

        print(f'Epoch {epoch+1} | Train loss: {train_loss[-1]:.4f} | Val loss: {val_loss[-1]:.4f}')

    model.load_state_dict(bestModel)

    return train_loss, val_loss

def test(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return accuracy_score(y_true, y_pred)

def plot_loss(train_loss, val_loss, title, pltshow = False, save_path=PATH_TO_PLOTS):
    plt.plot(train_loss, label=f'{title}-Train loss')
    plt.plot(val_loss, label=f'{title}-Val loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if pltshow:
        plt.show()
    plt.savefig(os.path.join(save_path, f'{title}.png'))
    plt.close()

def model_experiment(model, train_loader, val_loader, test_loader, epochs, optimizer, criterion, model_name, pltshow=False):
    train_loss, val_loss = train(model, train_loader, val_loader, epochs, optimizer, criterion)
    test_acc = test(model, test_loader)
    test_acc = test_acc * 100
    plot_loss(train_loss, val_loss, pltshow=pltshow,title=model_name)

    print(f'Test accuracy: {test_acc:.4f}')
    with open('./Lab4/test_accuracy.txt', 'a') as file:
        file.write(f'Test Accuracy of {model_name}: {test_acc:.3f}%\n')

def main():
    set_seed()
    epochs = 50
    batch_size = 4

    base_transform = transforms.Compose([transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    train_loader, val_loader, test_loader = get_dataloader(train_dataset_root='./mydataset/Train_resized',
                                                           val_dataset_root='./mydataset/Val_resized',
                                                           test_dataset_root='./mydataset/Test_resized',
                                                           batch_size=batch_size,
                                                           transform=base_transform)
    criterion = nn.CrossEntropyLoss()

    model1 = BaseCNN()
    optimizer = torch.optim.SGD(model1.parameters(), lr=0.001)
    model_experiment(model1, train_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer, criterion=criterion, model_name='BaseCNN')

    model2 = DropoutCNN()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001)
    model_experiment(model2, train_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer2, criterion=criterion, model_name='DropoutCNN')

    model3 = BatchNormCNN()
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.001)
    model_experiment(model3, train_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer3, criterion=criterion, model_name='BatchNormCNN')

    model4 = BaseCNN()
    optimizer4 = torch.optim.SGD(model4.parameters(), lr=0.001)
    augment_transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomRotation(degrees=(-90, 90)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    Atrain_dataset = datasets.ImageFolder(root='./mydataset/Train_resized', transform=augment_transform)
    Atrain_loader = DataLoader(Atrain_dataset, batch_size=batch_size, shuffle=False)
    model_experiment(model4, Atrain_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer4, criterion=criterion, model_name='BaseCNN-augment')

    model5 = BaseCNN()
    optimizer5 = torch.optim.SGD(model5.parameters(), lr=0.001)
    Strain_dataset = datasets.ImageFolder(root='./mydataset/Train_resized', transform=base_transform)
    Strain_loader = DataLoader(Strain_dataset, batch_size=batch_size, shuffle=True)
    model_experiment(model5, Strain_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer5, criterion=criterion, model_name='BaseCNN-shuffle')

    model6 = BaseCNN()
    optimizer6 = torch.optim.Adam(model6.parameters(), lr=0.0001)
    model_experiment(model6, train_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer6, criterion=criterion, model_name='BaseCNN-Adam')

    model7 = BestCNN()
    optimizer7 = torch.optim.Adam(model7.parameters(), lr=0.0001)
    AStrain_loader = DataLoader(Atrain_dataset, batch_size=batch_size, shuffle=True)
    model_experiment(model7, AStrain_loader, val_loader, test_loader, epochs=epochs, optimizer=optimizer7, criterion=criterion, model_name='BestCNN')


if __name__ == '__main__':
    main()