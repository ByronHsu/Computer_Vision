from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import numpy as np

def aug(mode = 'train'):
    if mode == 'train': 
        compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(48, scale = (0.8, 1)),
            transforms.RandomAffine(degrees = 15, translate = (0.1, 0.1), scale = (0.8, 1), shear = 15),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        return compose
    else:
        compose = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        return compose
 

class myset(Dataset):
    def __init__(self):
        path = 'data/'
        data, label = [], []
        for file in os.listdir(os.path.join(path, 'Real')):
            file_path = os.path.join(path, 'Real', file)
            data.append(cv2.imread(file_path, 0))
            label.append(0)

        for file in os.listdir(os.path.join(path, 'Synthetic')):
            if file.endswith('.pfm') == True: continue
            file_path = os.path.join(path, 'Synthetic', file)
            data.append(cv2.imread(file_path, 0))
            label.append(1)

        self.data = data
        self.label = label

    def __getitem__(self, i):
        img = np.expand_dims(self.data[i], axis = 2)
        T = aug('train')
        img = T(img)
        label = np.float32(self.label[i])
        return img, label

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    myset()
