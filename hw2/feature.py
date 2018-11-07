import torch
import matplotlib.pyplot as plt
import numpy as np
from p3 import LeNet5, ImageSet
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE

# detect gpu
use_cuda = torch.cuda.is_available()

model = LeNet5()
if use_cuda:
   model.cuda()
model.load_state_dict(torch.load('lenet5.pt'))
model.eval()


def visualize_feature(image, layer_idx):
   """
   convnet
   (c1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
   (relu1): ReLU()
   (s2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
   (c3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
   (relu2): ReLU()
   (s4): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
   """
   if use_cuda:
      image = image.cuda()
   output = image
   for index, layer in enumerate(model.convnet):
      output = layer(output)
      if index == layer_idx:
         return output

data = ImageSet('p3_data/valid/', [5000, 5100]) # 每組100張
data_loader = DataLoader(data, batch_size = 1)

def plot(X, title):
   em = TSNE(n_components = 2).fit_transform(X)
   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 100):
      x_em, y_em = em[i : i + 100, 0], em[i : i + 100, 1]
      ax.scatter(x_em, y_em, alpha = 0.8, edgecolors='none')
   plt.title('{}'.format(title))
   plt.savefig('{}.png'.format(title))
   print('Saving {}.png ...'.format(title))

def run():
   l_fs, h_fs = [], []
   for i, (image, label) in enumerate(data_loader):
      l_f = visualize_feature(image, 0)
      h_f = visualize_feature(image, 5)
      l_fs.append(l_f.detach().numpy())
      h_fs.append(h_f.detach().numpy()) # convert to np

   l_fs = np.array(l_fs, dtype = np.float_).reshape(len(l_fs), -1)
   h_fs = np.array(h_fs, dtype = np.float_).reshape(len(h_fs), -1)
   plot(l_fs, 'Low-Level Features')
   plot(h_fs, 'High-Level Features')

if __name__ == '__main__':
   run()

