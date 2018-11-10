import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from hw2_2_utils import process_data
from hw2_2_model import Model
from sklearn.neighbors import KNeighborsClassifier

o_prefix = 'hw2-2_output/'
H, W = 56, 46

class_list = list(range(1, 40 + 1))
train_list = list(range(1, 7 + 1))
test_list = list(range(8, 10 + 1))

def meanface():
   img = model.mean.reshape(H, W)
   cv2.imwrite(o_prefix + 'meanface.png', img)
   print('Saving meanface.png ...')

def first_five_eigenfaces():
   pcas = model.pcas
   for i in range(5):
      img = pcas[:, i]
      img = img - np.min(img) # 把最小值移回設0
      img = img * (255 / np.max(img)) # 把最大值擴為255
      img = img.reshape(H, W)
      cv2.imwrite(o_prefix + 'eigenface-' + str(i) + '.png', img)
      print('Saving eigenface-' + str(i) + '.png ...')

def reconstruct_pca(dim):
   pcas, mean = model.pcas, model.mean
   # pcas: 2576 x n, img: 2576
   i_prefix = 'hw2-2_data/'
   img = cv2.imread(i_prefix + '8_6.png', 0).flatten()

   y = model.project_to_pca(img.reshape(-1, 2576), dim)
   
   x = model.reconstruct_by_pca(y, dim).reshape(-1)

   rmse = np.sum((img - x) ** 2) / (H * W)
   print('n = {}, MSE = {}'.format(pcas.shape[1] ,rmse))
   x = x.reshape(H, W)
   o_prefix = 'hw2-2_output/'
   cv2.imwrite(o_prefix + '8_6' + '-n=' + str(pcas.shape[1]) + '.png', x)
   print('Saving ' + '8_6' + '-n=' + str(pcas.shape[1]) + '.png ...')

# visualize pca-projected images using tsne
def pca_projected_vis(imgs, dim): # imgs: testing data
   o_prefix = 'hw2-2_output/'
   # pcas: 2576 x dim, imgs: N x 2576 
   Y = model.project_to_pca(imgs, 100) # dim = 100
   em = TSNE(n_components = 2).fit_transform(Y)

   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em)
   plt.title('PCA Scattering DIM = 100')
   plt.savefig(o_prefix + 'PCA-scattering.png')
   print('Saving PCA-scattering.png ...')
   
def first_five_fisherfaces():
   pcas, ldas = model.pcas, model.ldas
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
def lda_projected_vis(imgs, dim): # imgs: testing data
   o_prefix = 'hw2-2_output/'
   Y = model.project_to_lda(imgs, dim)
   em = TSNE(n_components = 2).fit_transform(Y)
   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em)
   plt.title('LDA Scattering DIM = 30')
   plt.savefig(o_prefix + 'LDA-scattering.png')
   print('Saving LDA-scattering.png ...')

def cross_valid(k, n, Type):
   knn = KNeighborsClassifier(n_neighbors = k)
   if Type == 0:
      print('[KNN PCA] n : {} k = {}'.format(n, k))
   elif Type == 1:
      print('[KNN LDA] n : {} k = {}'.format(n, k))

   total_acc = 0
   for i in range(3):
      ratio = 2 / 3
      msk = np.random.rand(280) < ratio
      train_x, train_y = data_train[msk], data_train_label[msk]
      valid_x, valid_y = data_train[~msk], data_train_label[~msk]
      if Type == 0:
         train_p = model.project_to_pca(train_x, n) # projected
         valid_p = model.project_to_pca(valid_x, n)
      elif Type == 1:
         train_p = model.project_to_lda(train_x, n)
         valid_p = model.project_to_lda(valid_x, n)       
      knn.fit(train_p, train_y) # fit training data
      predict = knn.predict(valid_p)
      right = np.sum(np.equal(predict, valid_y)) # 算有幾個答對了
      acc = right / valid_p.shape[0] # 正確率
      total_acc += acc
   total_acc = total_acc / 3
   print('Validation Acc : {:.5f}'.format(total_acc))

def test(k, n, Type):
   knn = KNeighborsClassifier(n_neighbors = k)
   # verify on the testing set
   train_x, train_y = data_train, data_train_label
   test_x, test_y = data_test, data_test_label
   if Type == 0:
      train_p = model.project_to_pca(train_x, n) # projected
      test_p = model.project_to_pca(test_x, n)
   elif Type == 1:
      train_p = model.project_to_lda(train_x, n)
      test_p = model.project_to_lda(test_x, n)       
   knn.fit(train_p, train_y) # fit training data
   predict = knn.predict(test_p)
   right = np.sum(np.equal(predict, test_y)) # 算有幾個答對了
   acc = right / test_p.shape[0] # 正確率
   print('Testing Acc : {:.5f}'.format(acc))

model = Model()
data_train, data_train_label = process_data('hw2-2_data/', train_list)
data_test, data_test_label = process_data('hw2-2_data/', test_list)

if __name__ == '__main__':
   model.mean = np.average(data_train, axis = 0) # set mean
   pcas = model.fit_PCA(data_train)
   ldas = model.fit_LDA(data_train)
   # testcase a-1
   # meanface()
   # testcase a-2
   # first_five_eigenfaces()
   # testcase a-3
   # for i in [5, 50, 150, pcas.shape[1]]:
   #   reconstruct_pca(i)
   # testcase a-4
   # pca_projected_vis(data_test, 100)
   # testcase b-1
   # first_five_fisherfaces()
   # testcase b-2
   # lda_projected_vis(data_test, 30)
   # testcase c
   for t in [0, 1]:
      for n in [3, 10, 39]:
         for k in [1, 3, 5]:
            cross_valid(k, n, t)
            test(k, n, t)
            print()

   