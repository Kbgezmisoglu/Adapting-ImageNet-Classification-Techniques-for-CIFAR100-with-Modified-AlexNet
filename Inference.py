from torchvision.utils import *
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T
from PIL import Image
from transform import *
from model import *
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CLASS_LABELS import *

if __name__ == "__main__":
    QUICKNET_CHECKPOINT = "pth/SGD_pth/AlexNet_SGD_300.pth"

    # If you use GPU, you should write there "cuda" 
    # DEVICE = torch.device("cuda")

    # If you use MAC, you should write there "mps" 
    DEVICE = torch.device("mps")


    positives = 0
    negatives = 0

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 1

    testset = torchvision.datasets.CIFAR100(root='./', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    model = BuMuNet()

    # checkpoint = torch.load(QUICKNET_CHECKPOINT, map_location="cuda")
    checkpoint = torch.load(QUICKNET_CHECKPOINT, map_location="mps")

    model.load_state_dict(checkpoint)
    model = model.to(device=DEVICE)


    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        model.eval()

        # forward + backward + optimize
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        if labels[0].cpu().int() == predicted[0].cpu().int():
            positives += 1
        else:
            negatives += 1

    print(f"Accuracy: {positives / (positives + negatives)}")
