import numpy as np
import cv2
import os

class_list = list(range(1, 40 + 1))
train_list = list(range(1, 7 + 1))
test_list = list(range(8, 10 + 1))
# np.set_printoptions(threshold=np.inf) # 把np content全部print出

def process_data(prefix, idxs): # return n * m vector, n代表有幾張圖, m為flatten後的維度
   img_list = []
   label_list = []
   for i in class_list:
      for j in idxs:
         file_name = str(i) + '_' + str(j) + '.png'
         img = cv2.imread(os.path.join(prefix, file_name), 0)
         img = img.flatten()
         img_list.append(img)
         label_list.append(i)
   imgs = np.array(img_list, dtype = float)
   labels = np.array(label_list, dtype = float)
   return imgs, labels