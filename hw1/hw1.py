import cv2
import numpy as np
import matplotlib.pyplot as plt
i_prefix, o_prefix = 'testdata/', 'output/'

def conventional_bgr_to_y(bgr, file):
   cvt = np.array([0.0114, 0.0587, 0.299]).transpose() # b g r
   y = bgr.dot(cvt)
   folder = file + '/'
   file_name = 'c-' + file + '.png'
   cv2.imwrite(o_prefix + folder + file_name, y)
   print('Saving file ' + file_name + '...')
   
def quantize_bgr_to_y(bgr, file):
   L = [round(i * 0.1, 2) for i in range(11)]
   W_list = []
   # initilize the weighs
   for i in L:
      for j in L:
         if i + j <= 1:
            W_list.append((i, j, round(1 - i - j, 2)))
   W = np.array(W_list)

   folder = file + '/'

   for i in range(len(W)):
      cvt = W[i]
      y = bgr.dot(cvt)
      file_name = str(cvt[0]) + '-' + str(cvt[1]) + '-' + str(cvt[2]) + '-' + file + '.png'
      cv2.imwrite(o_prefix + folder + file_name, y)
      print('Saving file ' + file_name + '...')

   

if __name__ == '__main__':
   images = ['0a']
   img = cv2.imread(i_prefix + images[0] + '.png')
   conventional_bgr_to_y(img, images[0])
   quantize_bgr_to_y(img, images[0])