import numpy as np
import cv2

class Model:
   def __init__(self):
      self.mean = None
      self.pcas = None
      self.ldas = None
   def fit_PCA(self, imgs):
      cov = np.cov(imgs, rowvar = False) # 以column為feature, row為一單位
      e_vals, e_vecs = np.linalg.eig(cov) # e_vals會由大到小排列 且實數在前
      e_vecs = np.real(e_vecs)
      N = imgs.shape[0]
      self.pcas = e_vecs[:, 0 : N - 1] # 0 ~ N-2, 2576 X N - 1
      return self.pcas
   def fit_LDA(self, imgs, labels):
      pcas = self.pcas
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
      e_vecs = np.real(e_vecs)
      self.ldas = e_vecs[:, 0 : g_num - 1] # 0 ~ g_num - 2, (N - C) x (C - 1)
      return self.ldas
   def project_to_pca(self, imgs, dim): # 用幾根pca來project, 注意imgs維度是 N x 2576
      pcas = self.pcas[:, : dim]
      Y = (imgs - self.mean) @ pcas
      return Y
   def reconstruct_by_pca(self, project, dim): # project維度為 N x Pdim
      pcas = self.pcas[:, : dim]
      Y = project @ pcas.transpose() + self.mean
      return Y # 回傳 N x 2576的圖
   def project_to_lda(self, imgs, dim): # 用幾根lda來project, 注意imgs維度是 N x 2576
      N, C = 280, 40
      # 先投到pcas
      pcas = self.pcas[:, : N - C] # 切pca
      X = (imgs - self.mean) @ pcas # N x (N - C), 一個row代表一個投影
      # 再投到ldas
      ldas = self.ldas[:, : dim] # 切lda
      Y = X @ ldas # N x dim
      return Y