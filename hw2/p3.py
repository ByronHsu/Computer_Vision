import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

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

class ImageSet(Dataset):
   def __init__(self, folder, rng):
      # load all data into data_list
      data_list = []
      for i in range(10):
         for j in range(*rng):
            img_path = 'class_{}/{}.png'.format(i, str(j).zfill(4)) # zfill: 自動補齊4位0
            img = cv2.imread(folder + img_path, 0) # read as grey
            img = img.astype(np.float32)
            img = img.reshape(1, 28, 28) # img: 28 x 28
            img = img / 255 # normalize
            # print(img)
            # input()
            label = i
            data_list.append({'img': img, 'label': label})
      self.data = data_list
   def __getitem__(self, idx):
      # return img, label
      img = self.data[idx]['img']
      label = self.data[idx]['label']
      return img, label 
   def __len__(self):
      return len(self.data)

def train(e):
   net.train()
   epoch = 30
   for i, (images, labels) in enumerate(data_train_loader):
      if use_cuda:
         images, labels = images.cuda(), labels.cuda()
      optimizer.zero_grad()
      output = net(images)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      if i % 10 == 0:
         print('Train - Epoch {}, Batch: {}, Loss: {:3f}'.format(e, i, loss.data.item()))

def valid():
   net.eval()
   # validation set
   total_correct = 0
   avg_loss = 0.0
   for i, (images, labels) in enumerate(data_test_loader):
      if use_cuda:
         images, labels = images.cuda(), labels.cuda()
      output = net(images)
      avg_loss += criterion(output, labels).sum()
      pred = output.data.max(1)[1]
      total_correct += pred.eq(labels.data.view_as(pred)).sum()

   avg_loss /= len(data_test)
   print('Validation - Avg Loss: {:.3f}, Accuracy: {:.3f}'.format(avg_loss.data.item(), float(total_correct) / len(data_test)))
   
   # training set
   total_correct = 0
   avg_loss = 0.0
   for i, (images, labels) in enumerate(data_train_loader):
      if use_cuda:
         images, labels = images.cuda(), labels.cuda()
      output = net(images)
      avg_loss += criterion(output, labels).sum()
      pred = output.data.max(1)[1]
      total_correct += pred.eq(labels.data.view_as(pred)).sum()
   avg_loss /= len(data_train)
   print('Training - Avg Loss: {:.3f}, Accuracy: {:.3f}'.format(avg_loss.data.item(), float(total_correct) / len(data_train)))

def train_and_test(epoch):
   for i in range(epoch):
      train(i)
      valid()

net = LeNet5()
if use_cuda:
   net = net.cuda()
data_train = ImageSet('p3_data/train/', (0, 5000))
data_train_loader = DataLoader(data_train, batch_size = 256, shuffle = True)
data_test = ImageSet('p3_data/valid/', (5000, 6000))
data_test_loader = DataLoader(data_test, batch_size = 1024)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 1e-3)

if __name__ == '__main__':
   train_and_test(30)