import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
import pickle

i_prefix, g_prefix, f_prefix = 'testdata/', 'grey/', 'filter/'
if len(sys.argv) != 2:
   print('Usage: hw1-2.py [image_name]')

def gaussian_spatial(l, sigma):
   x = np.tile(range(l), (l, 1))
   x = x - l // 2
   y = x.transpose()
   index = - (x ** 2 + y ** 2) / (2 * (sigma ** 2))
   return np.exp(index)

def gaussian_range(window, sigma):
   l = window.shape[0]
   window = window.astype(float)
   center = window[l // 2, l // 2]
   a = (window - center)
   if len(window.shape) == 3:
      x = a[:, :, 0] ** 2 + a[:, :, 1] ** 2 + a[:, :, 2] ** 2
   else:
      x = a ** 2
   index = - x / (2 * (sigma ** 2))
   return np.exp(index)

def bilateral_filter(img, guide, sigma_s, sigma_r):
   h, w = img.shape[ : 2]
   window_s = 2 * 3 * sigma_s + 1
   pad = window_s // 2
   bgr = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
   if len(guide.shape) == 3:
      guide = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
   else:
      guide = np.pad(guide, ((pad, pad), (pad, pad)), 'symmetric')
   
   gs = gaussian_spatial(window_s, sigma_s)
   filtered = np.zeros((h, w, 3))

   for i in range(h):
      for j in range(w):
         window = guide[i : i + window_s, j : j + window_s]
         gr = gaussian_range(window, sigma_r)
         W = np.multiply(gs, gr).flatten()
         I = bgr[i : i + window_s, j : j + window_s].reshape(-1, 3)
         x = np.average(I, weights = W, axis = 0)
         filtered[i, j] = x
   return filtered

def vote():
   folder = img_name
   if not os.path.exists(f_prefix + folder):
      os.makedirs(f_prefix + folder)

   img = cv2.imread(i_prefix + img_name + '.png') # img
   sigma_s, sigma_r = [1, 2, 3], [12.75, 25.5, 51.0]
   vote_dic = {}
   
   for i in range(3):
      for j in range(3):
         one_vote(sigma_s[i], sigma_r[j], vote_dic)
         score_board = [(k, vote_dic[k]) for k in sorted(vote_dic, key = vote_dic.get, reverse = True)]
         
         print('Score_board:')
         for k in range(10):
            print('{}: {}'.format(score_board[k][0], score_board[k][1]))

   f = open('scoreboard/' + img_name + '.txt', 'w')
   f.write(str(score_board))

def one_vote(sigma_s, sigma_r, vote_dic):
   dirr = g_prefix + img_name
   dic = {}
   folder = img_name + '/' + str(sigma_s) + '-' + str(sigma_r)
   bf_img = bilateral_filter(img, img, sigma_s, sigma_r)

   if not os.path.exists(f_prefix + folder):
      os.makedirs(f_prefix + folder)

   for file_name in os.listdir(dirr):
      if not file_name.startswith('c-'):
         arr = [float(i) for i in file_name[0 : 10 + 1].split('-')]
         tup = (arr[0], arr[1], arr[2])
         
         guide = cv2.imread(g_prefix + img_name + '/' + file_name, 0)
         filtered = bilateral_filter(img, guide, sigma_s, sigma_r)
         delta = np.sum(np.absolute(bf_img - filtered))
         dic[tup] = delta
         
         cv2.imwrite(f_prefix + folder + '/' + file_name, filtered)
         print('Complete filtering {} with sigma_s = {} sigma_r = {}...'.format(file_name, sigma_s, sigma_r))
         print('Saving filtered image {}'.format(file_name))
         print('Delta = {}\n'.format(delta))


   file = open(f_prefix + folder + '/' + 'dic.pickle', 'wb')
   pickle.dump(dic, file)
   file.close()
   print('Save dic successfully!')

   arr = [(0.1, -0.1, 0), (-0.1, 0.1, 0), (0, 0.1, -0.1), (0, -0.1, 0.1), (0.1, 0, -0.1), (-0.1, 0, 0.1)]
   
   for key in dic:
   
      if key not in vote_dic:
         vote_dic[key] = 0
      
      is_local_min = True
      for a in arr:
         x = (round(key[0] + a[0], 2), round(key[1] + a[1], 2), round(key[2] + a[2], 2))
         if x in dic and dic[x] < dic[key]:
            is_local_min = False
            break

      if is_local_min == True:
         vote_dic[key] += 1

   return


img_name = sys.argv[1]
img = cv2.imread(i_prefix + img_name + '.png') # img
vote()