import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

training_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

label = 'idx,label\n'

for i in range(0, len(training_data)):
    img = ToPILImage()(training_data[i][0])
    img.save('./data/FashionMNIST/img/test_'+str(i)+'.jpg')
    label += str(i)+','+str(training_data[i][1])+'\n'
with open('./data/FashionMNIST/label/test.csv', 'w', encoding='utf-8') as f:
    f.write(label)