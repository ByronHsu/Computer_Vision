from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from .models import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

full_path = os.path.realpath(__file__)
folder, _ = os.path.split(full_path)

def evaluate(left, right):
    maxdisp = 64
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(**__imagenet_stats)])
    img_left = transform(left)
    img_right = transform(right)
    model = stackhourglass(maxdisp)
    model = nn.DataParallel(model, device_ids=[0])
    model.cuda()

    state_dict = torch.load(os.path.join(folder, 'checkpoint_4890.tar'))
    model.load_state_dict(state_dict['state_dict'])
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    model.eval()
    imgL = torch.FloatTensor(img_left).unsqueeze(0).cuda()
    imgR = torch.FloatTensor(img_right).unsqueeze(0).cuda()

    imgL, imgR = Variable(imgL), Variable(imgR)

    with torch.no_grad():
        output = model(imgL,imgR)

    output = torch.squeeze(output)
    pred_disp = output.data.cpu().numpy()

    return pred_disp
