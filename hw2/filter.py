import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import cv2
from p3 import LeNet5

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
   iter_times = 10000
   sig = nn.Sigmoid()
   for i in range(iter_times):
      y = torch.clamp(x, 0 ,1)
      optimizer.zero_grad()
      model(y)
      loss = - torch.mean(model.conv_output) + torch.mean(torch.abs(y)) # 加負 => gradient ascent
      print('Iter: {} Activation: {}'.format(i, loss.data.item()))
      loss.backward()
      optimizer.step()
   
   res = x.detach().numpy()
   res = res.reshape(28, 28)
   res = res * 255
   print(res.shape)
   cv2.imwrite('{}-{}.png'.format(layer_idx ,filter_idx), res)

# get_best_x(0, 1)
# get_best_x(0, 2)
print(model)
get_best_x(3, 1)
get_best_x(3, 2)
get_best_x(3, 3)
get_best_x(3, 4)
get_best_x(3, 5)
get_best_x(3, 6)