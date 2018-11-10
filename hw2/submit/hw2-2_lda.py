"""
• python3 hw2-2_lda.py $1 $2
• $1: path of whole dataset
• $2: path of the first 1 Fisherface
• E.g., python3 hw2-2_lda.py hw2-2_data hw2-2_output/output_fisher.png
"""
import numpy as np
import cv2
import sys
from hw2_2_utils import process_data
from hw2_2_model import Model

class_list = list(range(1, 40 + 1))
train_list = list(range(1, 7 + 1))
H, W = 56, 46
model = Model()
data_train, data_train_label = process_data(sys.argv[1], train_list)


def first_fisherfaces():
   pcas, ldas = model.pcas, model.ldas
   # fisherfaces = pcas @ ldas
   # d x (C - 1) = d x (N - C) @ (N - C) x (C - 1)
   d, N, C = 2576, 280, 40
   # slice pcas
   pcas = pcas[:, : N - C]
   fisherfaces = pcas @ ldas
   img = fisherfaces[:, 0]
   img = img - np.min(img) # 把最小值移回設0
   img = img * (255 / np.max(img)) # 把最大值擴為255
   img = img.reshape(H, W)
   cv2.imwrite(sys.argv[2], img)
   print('Saving image success!')

if __name__ == '__main__':
   model.mean = np.average(data_train, axis = 0) # set mean
   pcas = model.fit_PCA(data_train)
   ldas = model.fit_LDA(data_train)
   first_fisherfaces()
