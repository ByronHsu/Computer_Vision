import numpy as np
import cv2
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class_list = list(range(1, 40 + 1))
train_list = list(range(1, 7 + 1))
test_list = list(range(8, 10 + 1))
H, W = 56, 46

def process_train(): # return n * m vector, n代表有幾張圖, m為flatten後的維度
   img_list = []
   prefix = 'p2_data/'
   for i in class_list:
      for j in train_list:
         file_name = str(i) + '_' + str(j) + '.png'
         img = cv2.imread(prefix + file_name, 0)
         img = img.flatten()
         img_list.append(img)
   imgs = np.array(img_list, dtype = float)
   # print(img_np.shape)
   return imgs

def PCA(imgs):
   prefix = 'p2_output/'
   mean = np.average(imgs, axis = 0)
   img = mean.reshape(H, W)
   # cv2.imwrite(prefix + 'meanface.png', img)
   # print('Saving meanface.png ...')
   cov = np.cov(imgs, rowvar = False)
   e_vals, e_vecs = np.linalg.eig(cov)
   e_vecs = np.real(e_vecs) # 只取實部
   N = imgs.shape[0]
   # for i in range(5):
   #    img = e_vecs[:, i]
   #    img = img - np.min(img) # 把最小值移回設0
   #    img = img * (255 / np.max(img)) # 把最大值擴為255
   #    img = img.reshape(H, W)
   #    cv2.imwrite(prefix + 'eigenface-' + str(i) + '.png', img)
   #    print('Saving eigenface-' + str(i) + '.png ...')

   return e_vecs[:, 0 : N - 1] # 0 ~ N-2

def reconstruct_pca(pcas, imgs):
   # pcas: 2576 x A, img: 2576 x 1
   i_prefix = 'p2_data/'
   img = cv2.imread(i_prefix + '8_6.png', 0).flatten()
   mean = np.average(imgs, axis = 0)
   y = pcas.transpose() @ (img - mean) # A x 1
   x = pcas @ y + mean
   rmse = np.sum((img - x) ** 2) / (H * W)
   print('n = {}, MSE = {}'.format(pcas.shape[1] ,rmse))
   x = x.reshape(H, W)
   o_prefix = 'p2_output/'
   cv2.imwrite(o_prefix + '8_6' + '-n=' + str(pcas.shape[1]) + '.png', x)
   print('Saving ' + '8_6' + '-n=' + str(pcas.shape[1]) + '.png ...')
# TODO: color
def draw_pca_test(pcas, mean):
   img_list = []
   o_prefix = 'p2_output/'
   i_prefix = 'p2_data/'
   for i in class_list:
      for j in test_list:
         file_name = str(i) + '_' + str(j) + '.png'
         img = cv2.imread(i_prefix + file_name, 0)
         img = img.flatten()
         img_list.append(img)
   imgs = np.array(img_list, dtype = float)

   # pcas: 2576 x A, imgs: 1 x 2576 
   Y = (imgs - mean) @ pcas # N x A, 一個row代表一個投影
   em = TSNE(n_components = 2).fit_transform(Y)

   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em, alpha = 0.8, edgecolors='none')
   plt.title('PCA Scattering DIM = 100')
   plt.savefig(o_prefix + 'PCA-scattering.png')
   print('Saving PCA-scattering.png ...')

def LDA(imgs, pcas): # pcas: N - C x 2576
   g_item, g_num = 7, 40
   # project imgs to pcas (dim = N - C)
   # Slice pcas to N - C
   pcas = pcas[:, : imgs.shape[0] - g_num]
   # pcas: 2576 x N - C, imgs: N x 2576, Y: N x N - C
   Y = imgs @ pcas
   imgs = Y
   # perform LDA on imgs
   # compute Sw, Sb
   Sw, Sb = 0, 0
   u = np.average(imgs, axis = 0)
   for i in range(0, g_num * g_item, g_item):
      X = imgs[i: i + g_item] # slice a group
      ui = np.average(X, axis = 0)
      for j in range(X.shape[0]):
         xj = X[j]
         Sw += (xj - ui).reshape(-1, 1) @ (xj - ui).reshape(1, -1)
      Sb += (ui - u).reshape(-1, 1) @ (ui - u).reshape(1, -1)
   # find the eigenvectors of Sw.inv, Sb
   mat = np.linalg.inv(Sw) @ Sb
   e_vals, e_vecs = np.linalg.eig(mat)
   e_vecs = np.real(e_vecs) # 只取實部 N - C x N - C
   e_vecs = pcas @ e_vecs # 2576 x N - C
   # o_prefix = 'p2_output/'
   # for i in range(5):
   #    img = e_vecs[:, i]
   #    img = img - np.min(img) # 把最小值移回設0
   #    img = img * (255 / np.max(img)) # 把最大值擴為255
   #    img = img.reshape(H, W)
   #    cv2.imwrite(o_prefix + 'fisherface-' + str(i) + '.png', img)
   #    print('Saving fisherface-' + str(i) + '.png ...')
   return e_vecs[:, 0 : g_num - 1] # 0 ~ g_num - 2

# TODO: color
def draw_lda_test(ldas, mean):
   img_list = []
   o_prefix = 'p2_output/'
   i_prefix = 'p2_data/'
   for i in class_list:
      for j in test_list:
         file_name = str(i) + '_' + str(j) + '.png'
         img = cv2.imread(i_prefix + file_name, 0)
         img = img.flatten()
         img_list.append(img)
   imgs = np.array(img_list, dtype = float)

   # ldas: 2576 x A, imgs: 1 x 2576 
   Y = (imgs - mean) @ ldas # N x A, 一個row代表一個投影
   em = TSNE(n_components = 2).fit_transform(Y)

   fig, ax = plt.subplots()
   for i in range(0, em.shape[0], 3):
      x_em, y_em = em[i : i + 3, 0], em[i : i + 3, 1]
      ax.scatter(x_em, y_em, alpha = 0.8, edgecolors='none')
   plt.title('LDA Scattering DIM = 30')
   plt.savefig(o_prefix + 'LDA-scattering.png')
   print('Saving LDA-scattering.png ...')

def cross_validation(k, n, imgs, vecs): # vecs代表pcas/ldas
   labels = []
   l = imgs.shape[0] # 有幾個img
   g_num, g_item = 40, 7
   for i in range(g_num):
      for j in range(g_item):
         labels.append(i + 1) # img_1-40
   labels = np.array(labels, dtype = int)
   total_acc = 0
   # vecs 只取n cols
   vecs = vecs[:, :n]
   
   for i in range(3):
      ratio = 2 / 3
      msk = np.random.rand(g_item * g_num) < ratio
      # msk = np.ones(g_item, dtype = bool)
      # msk[round(g_item * (i / 3)) : round(g_item * ( (i + 1) / 3 ))] = False
      # msk = np.tile(msk, g_num)
      # slice出train, test
      train_x = imgs[msk]
      train_y = labels[msk]
      test_x = imgs[~msk]
      test_y = labels[~msk]

      # 得 train_p (projected) N x n
      train_p = train_x @ vecs
      # 得 test_p (projected) N x n
      test_p = test_x @ vecs
      correct = 0
      # 對每個test_p跑一次train_p找最近的k個, 看哪種train_y最多 則為那個種類
      # 將投影作KNN
      for m in range(test_p.shape[0]):
         dis = []
         Map = {}
         for j in range(train_p.shape[0]):
            # list stores tuple(label, distance), sort by distance
            dis.append((train_y[j], np.linalg.norm(test_p[m] - train_p[j])))
            # dict stores (key: occur times), sort by occur
         dis.sort(key = lambda tup: tup[1])
         res, val = 0, 0 # res: KNN決定出的label, val: 現在出現的最大次數
         for j in range(k):
            idx = dis[j][0]
            if idx not in Map:
               Map[idx] = 1
            else:
               Map[idx] += 1
            if Map[idx] > val:
               res = idx
               val = Map[idx]
         if res == test_y[m]:
            correct += 1
      # print('Accuracy: {}'.format(correct / test_p.shape[0]))
      total_acc += correct / test_p.shape[0]
   total_acc = total_acc / 3
   print('Total Accuracy: {}'.format(round(total_acc, 3)))

if __name__ == '__main__':
   imgs = process_train()
   pcas = PCA(imgs)
   # for i in [5, 50, 150, pcas.shape[1]]:
   #    reconstruct_pca(pcas[:, :i], imgs)
   # draw_pca_test(pcas[:, :100], np.average(imgs, axis = 0))
   ldas = LDA(imgs, pcas)
   # draw_lda_test(ldas[:, :30], np.average(imgs, axis = 0))
   K, N = [1, 3, 5], [3, 10, 39]

   for k in K:
      for n in N:
         print('PCA: k = {}, n = {}'.format(k, n))
         cross_validation(k, n, imgs, pcas)

   for k in K:
      for n in N:
         print('LDA: k = {}, n = {}'.format(k, n))
         cross_validation(k, n, imgs, ldas)
   