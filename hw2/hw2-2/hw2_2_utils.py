import numpy as np
import cv2

class_list = list(range(1, 40 + 1))
train_list = list(range(1, 7 + 1))
test_list = list(range(8, 10 + 1))

def process_data(prefix, idxs): # return n * m vector, n代表有幾張圖, m為flatten後的維度
   img_list = []
   for i in class_list:
      for j in idxs:
         file_name = str(i) + '_' + str(j) + '.png'
         img = cv2.imread(prefix + file_name, 0)
         img = img.flatten()
         img_list.append(img)
   imgs = np.array(img_list, dtype = float)
   # print(img_np.shape)
   return imgs

def PCA(imgs):
   cov = np.cov(imgs, rowvar = False) # 以column為feature, row為一單位
   e_vals, e_vecs = np.linalg.eig(cov)
   sort_idxs = np.argsort(e_vals)[::-1] # 由大到小排序, 回傳indexs
   e_vecs = e_vecs[:, sort_idxs] # 排序e_vecs
   e_vecs = np.real(e_vecs)
   e_vals = e_vals[sort_idxs]
   N = imgs.shape[0]
   return e_vecs[:, 0 : N - 1] # 0 ~ N-2, 2576 X N - 1

def LDA(imgs):
   pcas = PCA(imgs)
   g_item, g_num = 7, 40
   # Slice pcas to N - C
   pcas = pcas[:, : imgs.shape[0] - g_num]
   # project imgs to pcas (dim = N - C)
   # pcas: 2576 x N - C, imgs: N x 2576, Y: N x N - C
   Y = (imgs - np.average(imgs, axis = 0)) @ pcas
   imgs = Y
   # perform LDA on imgs
   # compute Sw, Sb
   Sw, Sb = 0, 0
   u = np.average(imgs, axis = 0) # 全部圖片的平均
   for i in range(0, g_num * g_item, g_item):
      X = imgs[i: i + g_item] # slice a group
      ui = np.average(X, axis = 0) # 這組的平均
      for j in range(X.shape[0]):
         xj = X[j]
         Sw += (xj - ui).reshape(-1, 1) @ (xj - ui).reshape(1, -1)
      Sb += (ui - u).reshape(-1, 1) @ (ui - u).reshape(1, -1)
   # find the eigenvectors of Sw.inv, Sb
   mat = np.linalg.inv(Sw) @ Sb
   e_vals, e_vecs = np.linalg.eig(mat)
   sort_idxs = np.argsort(e_vals)[::-1] # 由大到小排序, 回傳indexs
   e_vecs = e_vecs[:, sort_idxs] # 排序e_vecs
   e_vecs = np.real(e_vecs)
   e_vals = e_vals[sort_idxs]
   return e_vecs[:, 0 : g_num - 1] # 0 ~ g_num - 2, (N - C) x (C - 1)