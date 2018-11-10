"""
• python3 hw2-3_train.py $1
• $1: directory of the hw2-3_data folder
• E.g., python3 hw2-3_train.py ./hw2/hw2-3_data/
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from hw2_3_model import LeNet5
from hw2_3_utils import ImageSet

o_prefix = 'hw2-3_output/'
# detect gpu
use_cuda = torch.cuda.is_available()

def train(e):
   net.train()
   for i, (images, labels) in enumerate(data_train_loader):
      if use_cuda:
         images, labels = images.cuda(), labels.cuda()
      optimizer.zero_grad()
      output = net(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      if i % 10 == 0:
         print('Train - Epoch {}, Batch: {}, Loss: {:.5f}'.format(e, i, loss.data.item()))

def valid():
   net.eval()
   # validation set
   total_correct = 0
   total_loss = 0.0
   for i, (images, labels) in enumerate(data_test_loader):
      if use_cuda:
         images, labels = images.cuda(), labels.cuda()
      output = net(images)
      total_loss += criterion(output, labels).sum()
      pred = output.data.max(1)[1]
      total_correct += pred.eq(labels.data.view_as(pred)).sum()

   avg_loss = total_loss * data_test_loader.batch_size / len(data_test) # 一個loss代表batch中平均的loss
   print('Validation - Avg Loss: {:.5f}, Accuracy: {:.5f}'.format(avg_loss.data.item(), float(total_correct) / len(data_test)))
   
   # training set
   total_correct = 0
   total_loss = 0.0
   for i, (images, labels) in enumerate(data_train_loader):
      if use_cuda:
         images, labels = images.cuda(), labels.cuda()
      output = net(images)
      total_loss += criterion(output, labels).sum()
      pred = output.data.max(1)[1]
      total_correct += pred.eq(labels.data.view_as(pred)).sum()

   acc = float(total_correct) / len(data_train)
   avg_loss = total_loss * data_train_loader.batch_size / len(data_train)
   train_acc.append(acc)
   train_loss.append(avg_loss.data.item())
   print('Training - Avg Loss: {:.5f}, Accuracy: {:.5f}'.format(avg_loss.data.item(), acc))

train_acc = []
train_loss = []

def plot():
   plt.subplots_adjust(wspace = 0.5)
   plt.subplot(1,2,1) # row 1 col 2 畫在第一個
   plt.xlabel('Iterations')
   plt.ylabel('Training accuracy')
   plt.title('Learning Curve')
   plt.plot(train_acc)
   plt.subplot(1,2,2)
   plt.xlabel('Iterations')
   plt.ylabel('Training loss')
   plt.title('Learning Curve')
   plt.plot(train_loss)
   plt.savefig(os.path.join(o_prefix, 'learning-curve.png'))
   print('Successfully save learning curve png!')

def train_and_valid(epoch):
   for i in range(epoch):
      train(i)
      # valid()
   # plot()


net = LeNet5()
if use_cuda:
   net = net.cuda()

data_train = ImageSet(os.path.join(sys.argv[1],'train/'), (0, 5000))
data_train_loader = DataLoader(data_train, batch_size = 256, shuffle = True)
data_test = ImageSet(os.path.join(sys.argv[1],'valid/'), (5000, 6000))
data_test_loader = DataLoader(data_test, batch_size = 1024)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-3)

if __name__ == '__main__':
   train_and_valid(10)
   torch.save(net.state_dict(), 'lenet5.pt')
