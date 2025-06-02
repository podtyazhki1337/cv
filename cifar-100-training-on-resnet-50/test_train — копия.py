import comet_ml
import torch # pytorch basic package
from torch import nn # neural net
import gc
from torch.utils.data import DataLoader, Dataset # to work with data
from torchvision import datasets # built-in data
from torchvision.transforms import ToTensor # to convert nparrays/images into pytorch tensors
from torchvision import models
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from nn_utils import train, test

import random
import os
from PIL import Image, ImageDraw, ImageFont

from comet_ml import Experiment
import torchvision.transforms as transforms



experiment = Experiment(
  api_key="Yb0Ffy5WaM3qRxJ3qZAjvcboV",
  project_name="test",
  workspace="podtyazhki1337"
)


experiment.set_name("test_resnet_to_upload_mod")

torch.manual_seed(0) # for reproducibility
device = "cuda" if torch.cuda.is_available() else "cpu" # set GPU
print(device)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Download test data from open datasets.
test_data = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=transform_test
)

# Download training data from open datasets.
training_data = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=transform_train,
)

trainloader = DataLoader(
    training_data,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

testloader = DataLoader(
    test_data,
    batch_size=1024,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

num_classes = 100
epochs = 5
loss = nn.CrossEntropyLoss()



resnet_model_pretrained = models.resnet50(weights="IMAGENET1K_V1")
resnet_model_pretrained.fc = nn.Linear(in_features = resnet_model_pretrained.fc.in_features, out_features = num_classes)

resnet_model_pretrained = resnet_model_pretrained.to(device)



optimizer = torch.optim.AdamW(resnet_model_pretrained.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

train_losses = []
test_losses = []
running_corrects = 0
for t in tqdm(range(epochs)):
    print(f'EPOCH {t+1} --------------------')
    train_loss, training_accuracy = train(trainloader, resnet_model_pretrained, loss, optimizer, 200, device)
    test_loss, testing_accuracy = test(t, testloader, resnet_model_pretrained, loss, device)

    train_losses.append(train_loss)
    test_losses.append(test_loss)


    experiment.log_metric("train_loss", train_loss, epoch=t + 1)
    experiment.log_metric("train_accuracy", training_accuracy*100, epoch=t + 1)
    experiment.log_metric("val_loss",test_loss, epoch=t + 1)
    experiment.log_metric("val_accuracy", testing_accuracy*100, epoch=t + 1)
    scheduler.step()
torch.save(resnet_model_pretrained.state_dict(), "resnet_50_trained_cifar100.pth")
experiment.log_model("test_resnet_to_upload_mod", "resnet_50_trained_cifar100.pth")
print("Done!")



labels_map = {i: name for i, name in enumerate(test_data.classes)}
subset_indices = random.sample(range(len(test_data)), 10)
subset = torch.utils.data.Subset(test_data, subset_indices)
loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

os.makedirs("correct", exist_ok=True)
os.makedirs("incorrect", exist_ok=True)
try:
    font = ImageFont.truetype("arial.ttf", 8)
except:
    font = ImageFont.load_default()
counter = 0
for idx, (img_tensor, label) in enumerate(loader):
    img_tensor = img_tensor.to(device)
    label = label.to(device)

    with torch.no_grad():
        output = resnet_model_pretrained(img_tensor)
        pred = output.argmax(dim=1)

    gt_label = labels_map[label.item()]
    pred_label = labels_map[pred.item()]
    if gt_label == pred_label:
        counter+=0

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()].item()
    img_np = img_tensor.squeeze(0).cpu()
    img_np = img_np * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_np = img_np + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)


    img_pil = transforms.ToPILImage()(img_np)

    draw = ImageDraw.Draw(img_pil)
    text = f"GT: {gt_label} | Pred: {pred_label} P=: {prob*100:.2f}%"
    draw.rectangle([(0, 0), (224, 20)], fill=(255, 255, 255))
    draw.text((2, 2), text, font=font, fill=(0, 0, 0))


    out_dir = "correct" if pred.item() == label.item() else "incorrect"
    img_pil.save(os.path.join(out_dir, f"img_{idx:04d}.jpg"))

print("Done — images saved to ./correct and ./incorrect")