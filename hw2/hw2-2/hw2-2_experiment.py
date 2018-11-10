import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from hw2_2_utils import process_data, PCA, LDA
o_prefix = 'hw2-2_output/'
H, W = 56, 46

class_list = list(range(1, 40 + 1))
train_list = list(range(1, 7 + 1))
test_list = list(range(8, 10 + 1))

def meanface(imgs):
   mean = np.average(imgs, axis = 0)
   img = mean.reshape(H, W)
   cv2.imwrite(o_prefix + 'meanface.png', img)
   print('Saving meanface.png ...')

def first_five_eigenfaces(pcas):
   for i in range(5):
      img = pcas[:, i]
      img = img - np.min(img) # 把最小值移回設0
      img = img * (255 / np.max(img)) # 把最大值擴為255
      img = img.reshape(H, W)
      cv2.imwrite(o_prefix + 'eigenface-' + str(i) + '.png', img)
      print('Saving eigenface-' + str(i) + '.png ...')

def reconstruct_pca(n, pcas, mean):
   # 用n根來投影
   pcas = pcas[:, :n]
   # pcas: 2576 x n, img: 2576
   i_prefix = 'hw2-2_data/'
   img = cv2.imread(i_prefix + '8_6.png', 0).flatten()
   y = pcas.transpose() @ (img - mean) # n
   x = pcas @ y + mean
   rmse = np.sum((img - x) ** 2) / (H * W)
   print('n = {}, MSE = {}'.format(pcas.shape[1] ,rmse))
   x = x.reshape(H, W)
   o_prefix = 'hw2-2_output/'
   cv2.imwrite(o_prefix + '8_6' + '-n=' + str(pcas.shape[1]) + '.png', x)
   print('Saving ' + '8_6' + '-n=' + str(pcas.shape[1]) + '.png ...')

# visualize pca-projected images using tsne
def pca_projected_vis(imgs, mean, pcas): # imgs: testing data
   # pca dim = 100
   pcas = pcas[:, :100]
   o_prefix = 'hw2-2_output/'
   # pcas: 2576 x dim, imgs: N x 2576 
   Y = (imgs - mean) @ pcas # N x dim, 一個row代表一個投影
   em = TSNE(n_components = 2).fit_transform(Y)

   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em)
   plt.title('PCA Scattering DIM = 100')
   plt.savefig(o_prefix + 'PCA-scattering.png')
   print('Saving PCA-scattering.png ...')
   
def first_five_fisherfaces(ldas, pcas):
   # fisherfaces = pcas @ ldas
   # d x (C - 1) = d x (N - C) @ (N - C) x (C - 1)
   d, N, C = 2576, 280, 40
   # slice pcas
   pcas = pcas[:, : N - C]
   fisherfaces = pcas @ ldas
   for i in range(5):
      img = fisherfaces[:, i]
      img = img - np.min(img) # 把最小值移回設0
      img = img * (255 / np.max(img)) # 把最大值擴為255
      img = img.reshape(H, W)
      cv2.imwrite(o_prefix + 'fisherface-' + str(i) + '.png', img)
      print('Saving fisherface-' + str(i) + '.png ...')


# visualize lda-projected images using tsne
def lda_projected_vis(imgs, mean, ldas, pcas): # imgs: testing data
   o_prefix = 'hw2-2_output/'
   d, N, C = 2576, 280, 40
   dim = 39
   # 先投到pcas
   pcas = pcas[:, : N - C] # 切pca
   X = (imgs - mean) @ pcas # N x (N - C), 一個row代表一個投影
   # 再投到ldas
   ldas = ldas[:, : dim] # 切lda
   Y = X @ ldas # N x dim
   print(Y.shape)
   input()
   em = TSNE(n_components = 2).fit_transform(Y)

   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em)
   plt.title('LDA Scattering DIM = 30')
   plt.savefig(o_prefix + 'LDA-scattering.png')
   print('Saving LDA-scattering.png ...')

if __name__ == '__main__':
   data_train = process_data('hw2-2_data/', train_list)
   data_test = process_data('hw2-2_data/', test_list)
   mean = np.average(data_train, axis = 0)
   pcas = PCA(data_train)
   ldas = LDA(data_train)
   # test a-1
   # meanface(data_train)
   # test a-2
   # first_five_eigenfaces(pcas)
   # test a-3
   # for i in [5, 50, 150, pcas.shape[1]]:
   #    reconstruct_pca(i, pcas, mean)
   # test a-4
   # pca_projected_vis(data_test, mean, pcas)
   # test b-1
   # first_five_fisherfaces(ldas, pcas)
   # test b-2
   lda_projected_vis(data_test, mean, ldas, pcas)