"""
• python3 hw2-3_test.py $1 $2
• $1: directory of the testing images folder
• $2: path of the output prediction file
• E.g., python3 hw2-3_test.py ./test_images/ ./output.csv
• Testing images folder include images named:
• 0000.png , 0002.png , ... , 9999.png
• Output prediction file format • In csv format
• First row: “id,label”
• From second row: “<image_id>, <predicted_label>”
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hw2_3_utils import TestSet
from hw2_3_model import LeNet5
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

model = LeNet5()
model.load_state_dict(torch.load('lenet5.pt')) # remap everything onto CPU
model.eval()

data = TestSet(sys.argv[1]) # 
data_loader = DataLoader(data, batch_size = 1)

def test():
   ids = []
   for i in range(len(data)):
      ids.append(str(i).zfill(4))
   labels = []
   for i, images in enumerate(data_loader):
      y = model(images)
      y = torch.argmax(y).item()
      labels.append(y)
   res = pd.DataFrame({"id": ids, "label": labels})
   res.to_csv(sys.argv[2], index = False)
   print('Successfully save ans csv!')

if __name__ == '__main__':
   test()