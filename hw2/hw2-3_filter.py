import os
import cv2
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from p3 import LeNet5
o_prefix = 'hw2-3_output/'
# detect gpu
use_cuda = torch.cuda.is_available()

model = LeNet5()
if use_cuda:
   model.cuda()
model.load_state_dict(torch.load('lenet5.pt'))
model.eval()

def get_best_x(layer_idx, filter_idx):
   x = np.float32(np.random.uniform(0, 1, (1, 1, 28, 28))) # normalize過之圖
   x = torch.from_numpy(x)
   x.requires_grad = True

   def hook(module, In, out):
      # 每次跑到指定layer就會更新conv_output
      # In, out: len為1的tuple, In存進layer的array, out存出layer的array
      model.conv_output = out[0][filter_idx]

   model.convnet[layer_idx].register_forward_hook(hook)
   optimizer = optim.SGD([x], lr = 1)
   iter_times = 1000
   sig = nn.Sigmoid()
   for i in range(iter_times):
      y = torch.clamp(x, 0 ,1)
      optimizer.zero_grad()
      model(y)
      loss = - torch.mean(model.conv_output) # + torch.mean(torch.abs(y)) # 加負 => gradient ascent
      print('Iter: {} Activation: {}'.format(i, loss.data.item()))
      loss.backward()
      optimizer.step()
   
   res = x.detach().numpy()
   res = res.reshape(28, 28)
   res = res * 255
   print(res.shape)
   cv2.imwrite(os.path.join(o_prefix, '{}-{}.png'.format(layer_idx ,filter_idx)), res)

def run():
   for i in [0, 3]:
      for j in range(6):
         get_best_x(i, j)

if __name__ == '__main__':
   run()
