import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# detect gpu
use_cuda = torch.cuda.is_available()

class LeNet5(nn.Module): # Module: base class for nn model
   """
   Input - 28 x 28
   C1 - 6 @ 24 x 24(kernel : 5 x 5)
   relu1
   S2 - 6 @ 12 x 12
   C3 - 16 @ 8 x 8
   relu2
   S4 - 16 @ 4 x 4
   View - 256 

   F5 - 120
   relu3
   F6 - 84
   relu4
   F7 - 10
   output
   """
   def __init__(self):
      super(LeNet5, self).__init__()
      self.convnet = nn.Sequential(OrderedDict([
         ('c1', nn.Conv2d(1, 6, kernel_size = (5, 5))),
         ('relu1', nn.ReLU()),
         ('s2', nn.MaxPool2d(kernel_size = (2, 2), stride = 2)),
         ('c3', nn.Conv2d(6, 16, kernel_size = (5, 5))),
         ('relu2', nn.ReLU()),
         ('s4', nn.MaxPool2d(kernel_size = (2, 2), stride = 2)),
      ]))

      self.fc = nn.Sequential(OrderedDict([
         ('f5', nn.Linear(256, 120)),
         ('relu3', nn.ReLU()),
         ('f6', nn.Linear(120, 84)),
         ('relu4', nn.ReLU()),
         ('f7', nn.Linear(84, 10)),
         ('output', nn.Softmax(dim = 1))
      ]))
   def forward(self, img):
      output = self.convnet(img)
      output = output.view(-1, 256)
      output = self.fc(output)
      return output