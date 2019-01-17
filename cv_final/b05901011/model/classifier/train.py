from .model import net
import torch
import torch.optim as optim
import os
from .loader import myset, aug
from torch.utils.data import DataLoader

use_cuda = torch.cuda.is_available()
model = net()

if use_cuda:
   model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr = 1e-4)
criterion = torch.nn.BCELoss()

train_data = myset()
train_data_loader = DataLoader(train_data, batch_size = 10, shuffle = True)

def binary_accuracy(preds, y):
   rounded_preds = torch.round(preds)
   correct = (rounded_preds == y).float()
   acc = correct.sum()/len(correct)
   return acc

def train():
   epoch_loss = 0.0
   epoch_acc = 0.0 
   model.train()
   for i, (imgs, labels) in enumerate(train_data_loader):
      if use_cuda:
         imgs, labels = imgs.cuda(), labels.cuda()
      optimizer.zero_grad()
      out = model(imgs).squeeze(1)
      loss = criterion(out, labels)
      acc = binary_accuracy(out, labels)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item() * imgs.shape[0]
      epoch_acc += acc.item() * imgs.shape[0]
   l = len(train_data)
   print('Training - loss : {} acc : {}'.format(epoch_loss / l, epoch_acc / l))


if __name__ == '__main__':
    for e in range(1000):
        train()
    torch.save(model.state_dict(), os.path.join('models', 'net.pt'))
