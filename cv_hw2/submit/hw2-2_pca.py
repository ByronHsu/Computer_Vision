"""
• python3 hw2-2_pca.py $1 $2 $3
• $1: path of whole dataset
• $2: path of the input testing image
• $3: path of the output testing image reconstruct by all eigenfaces
• E.g., python3 hw2-2_pca.py hw2-2_data hw2-2_data/1_1.png hw2-2_output/output_pca.png
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

def reconstruct_pca(dim):
   pcas, mean = model.pcas, model.mean
   # pcas: 2576 x n, img: 2576
   img = cv2.imread(sys.argv[2], 0).flatten()
   y = model.project_to_pca(img.reshape(-1, 2576), dim)
   x = model.reconstruct_by_pca(y, dim).reshape(-1)
   x = x.reshape(H, W)
   cv2.imwrite(sys.argv[3], x)
   print('Saving image success!')


if __name__ == '__main__':
   model.mean = np.average(data_train, axis = 0) # set mean
   pcas = model.fit_PCA(data_train)
   reconstruct_pca(279)
