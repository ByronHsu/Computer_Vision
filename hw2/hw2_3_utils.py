import cv2
import os
import numpy as np
from torch.utils.data import Dataset

# TODO: Define transformer
class ImageSet(Dataset):
   def __init__(self, folder, rng):
      # load all data into data_list
      data_list = []
      for i in range(10):
         for j in range(*rng):
            img_path = 'class_{}/{}.png'.format(i, str(j).zfill(4)) # zfill: 自動補齊4位0
            img = cv2.imread(os.path.join(folder, img_path), 0) # read as grey
            img = img.astype(np.float32)
            img = img.reshape(1, 28, 28) # img: 28 x 28
            img = img / 255 # normalize
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

class TestSet(Dataset):
   def __init__(self, folder, rng):
      # load all data into data_list
      data_list = []
      for i in range(*rng):
         img_path = '{}.png'.format(str(i).zfill(4)) # zfill: 自動補齊4位0
         img = cv2.imread(os.path.join(folder, img_path), 0) # read as grey
         img = img.astype(np.float32)
         img = img.reshape(1, 28, 28) # img: 28 x 28
         img = img / 255 # normalize
         data_list.append({'img': img})
      self.data = data_list
   def __getitem__(self, idx):
      # return img
      img = self.data[idx]['img']
      return img
   def __len__(self):
      return len(self.data)