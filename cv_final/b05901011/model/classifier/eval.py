import numpy as np
import torch
import cv2
import sys
import os
from .model import net
from torchvision import transforms

def preprocess(img):
    img = np.expand_dims(img, axis = 2)
    compose = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return compose(img).unsqueeze(0)
# detect cuda
use_cuda = torch.cuda.is_available()

model = net()

full_path = os.path.realpath(__file__)
folder, _= os.path.split(full_path)

model.load_state_dict(torch.load(os.path.join(folder, 'checkpoints', 'pretrain.pt')))

if use_cuda == True:
    model = model.cuda()

def evaluate(img):
    # img is a np array, and it can only have 1 channel.
    input = preprocess(img)
    if use_cuda == True:
        input = input.cuda()
    return model(input)

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1], 0)
    print(eval(img))
