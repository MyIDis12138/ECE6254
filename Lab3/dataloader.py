import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# Define a transform to resize images to 224x224 and convert them to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder(root='./mydataset/Train_resized', transform=transform)

# Loaders for dataset, which allow to iterate through the dataset in batches
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

class_names = {v: k for k, v in train_dataset.class_to_idx.items()}
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))  # convert from Tensor image and rearrange dimensions to (height, width, channels)
    plt.imshow(img)
    plt.axis('off')  # To not display axes for better clarity

dataiter = iter(train_loader)
images, labels = next(dataiter)
fig = plt.figure(figsize=(8, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    imshow(images[i])
    plt.title(class_names[labels[i].item()])
plt.show()