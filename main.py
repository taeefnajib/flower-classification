'''
Code taken from: https://jovian.ai/mohanavignesh/flower-classification
'''

# Importing all dependencies
import torch
import torchvision
import torchvision
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Optional, List, Dict

# Creating dataclass
@dataclass_json
@dataclass
class Hyperparameters(object):
    base_dir: str = "flowers"
    validation_size: int = 600
    batch_size: int = 32
    num_epochs: int = 10
    opt_func: Optional[float] = torch.optim.Adam
    lr: float = 0.001


# Instantiating Hyperparameters class
hp = Hyperparameters()


# Creating dataset
def create_dataset(base_dir, validation_size):
    transformer = torchvision.transforms.Compose(
        [  # Applying Augmentation
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    dataset = ImageFolder(base_dir, transform=transformer)
    validation_size = validation_size
    training_size = len(dataset) - validation_size
    train_ds, val_ds_main = random_split(dataset, [training_size, validation_size])
    val_ds, test_ds = random_split(val_ds_main, [400, 200])
    return dataset, train_ds, val_ds, test_ds


# Specifying dataset, train_ds, val_ds and test_ds
dataset, train_ds, val_ds, test_ds = create_dataset(hp.base_dir, hp.validation_size)

# Loading data
def load_data(train_ds, val_ds, test_ds, batch_size):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, val_dl, test_dl


# Specifying train_dl, val_dl, test_dl
train_dl, val_dl, test_dl = load_data(train_ds, val_ds, test_ds, hp.batch_size)

# Checking accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# Building model
class FlowerClassificationModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, xb):
        return self.network(xb)

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


# Instantiating the class FlowerClassificationModel()
model = FlowerClassificationModel()

# Evaluating model
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# Fitting model
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Getting default device
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# Moving tensor to device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Loading data to device
class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def wf(
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    model: FlowerClassificationModel,
    num_epochs: int,
    lr: int,
    opt_func: Optional[float],
) -> List[Dict[str, float]]:
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(model, device)
    model = to_device(FlowerClassificationModel(), device)
    evaluate(model, val_dl)
    return fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


if __name__ == "__main__":
    wf(train_dl, val_dl, model, hp.num_epochs, hp.lr, hp.opt_func)
