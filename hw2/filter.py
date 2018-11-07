import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from p3 import LeNet5

# detect gpu
use_cuda = torch.cuda.is_available()

model = LeNet5()
if use_cuda:
   model.cuda()
model.load_state_dict(torch.load('lenet5.pt'))
model.eval()

class FilterVis():
   def __init__(self):
      self.x = 0
      self.conv_output = 0
   def get_best_x(self, layer_idx, filter_idx):
      self.x = np.float32(np.random.uniform(0, 1, (1, 1, 28, 28))) # normalize過之圖
      self.x = torch.from_numpy(self.x)

      def hook(module, In, out):
         # 每次跑到指定layer就會更新conv_output
         # In, out: len為1的tuple, In存進layer的array, out存出layer的array
         self.conv_output = out[0][filter_idx]

      model.convnet[layer_idx].register_forward_hook(hook)

      optimizer = optim.Adam([self.x], lr = 0.1)
      epoch = 30

      for _ in range(epoch):
         optimizer.zero_grad()
         model(self.x)
         loss = - torch.sum(self.conv_output) # 加負 gradient ascent
         print(loss.data.item())
         loss.backward()
         optimizer.step()
      
v = FilterVis()
v.get_best_x(0, 1)