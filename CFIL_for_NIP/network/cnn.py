import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .base import BaseNetwork

torch.backends.cudnn.benchmark = True
class CNN(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),


        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128*8*8, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=3)
            )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_transform = transforms.Compose([
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    validation_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

    nane = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    data_iter = iter(train_dataloader)
    img, labels = data_iter.next()

    model = CNN(10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    num_epochs = 15
    losses = []
    val_losses = []
    accs = []
    val_accs = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for imgs, labels in train_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            running_acc += torch.mean(pred.eq(labels).float())
            optimizer.step()
            running_loss /= len(train_dataloader)
            running_acc /= len(train_dataloader)
            losses.append(running_loss)
            accs.append(running_acc)

        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_imgs, val_labels in validation_dataloader:
            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)
            val_output = model(val_imgs)
            val_loss = criterion(val_output, val_labels)
            val_loss.backward()
            val_running_loss += val_loss.item()
            val_pred = torch.argmax(val_output, dim=1)
            val_running_acc += torch.mean(val_pred.eq(val_labels).float())
            optimizer.step()
            val_running_loss /= len(validation_dataloader)
            val_running_acc /= len(validation_dataloader)
            val_losses.append(val_running_loss)
            val_accs.append(val_running_acc)
            print("epoch: {}, los: {}, acc: {}, val loss: {}, val acc: {}".format(epoch, running_loss, running_acc,
                                                                                  val_running_loss, val_running_acc))

    params = model.state_dict()
    torch.save(params, "model.prm")