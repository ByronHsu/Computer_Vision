import os
import numpy as np
import cv2


# transform point to 2 x 9 (part of A)
# u: point on img, v: point on canvas
# return list
def point_to_vec(u, v):
   vecs = [
      [u[0], u[1], 1, 0, 0, 0, -u[0] * v[0], -u[1] * v[0], -v[0]],
      [0, 0, 0, u[0], u[1], 1, -u[0] * v[1], -u[1] * v[1], -v[1]]
   ]
   return vecs

class Homography:
   def __init__(self):
      pass
   
   def read_file(self):
      canvas_name = 'times_square.jpg'
      imgs_name = ['wu.jpg', 'ding.jpg', 'yao.jpg', 'kp.jpg', 'lee.jpg']
      canvas_corners = [
         np.array([[352, 818], [407, 818], [352, 884], [408, 885]]),
         np.array([[14, 311], [152, 157], [150, 402], [315, 278]]),
         np.array([[674, 364], [864, 279], [725, 430], [885, 369]]),
         np.array([[495, 808], [609, 802], [495, 892], [609, 896]]),
         np.array([[608, 1024], [664, 1032], [593, 1118], [651, 1134]])
      ]
      imgs_corners = []
      canvas = cv2.imread(os.path.join('input', canvas_name))
      imgs = []
      for i in range(5):
         img = cv2.imread(os.path.join('input', imgs_name[i]))
         corner = np.array([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
         imgs.append(img)
         imgs_corners.append(corner)
      self.canvas, self.imgs, self.canvas_corners, self.imgs_corners = canvas, imgs, canvas_corners, imgs_corners
   
   def solve_homography(self, k): # 第k張圖
      A = []
      for i in range(4):
         t = point_to_vec(self.imgs_corners[k][i], self.canvas_corners[k][i])
         A.append(t[0])
         A.append(t[1])
      A = np.array(A)
      u, s, vh = np.linalg.svd(A.transpose() @ A, full_matrices = False)
      Fn = u[:, 8].reshape(3, 3)
      H, W = self.imgs[k].shape[0], self.imgs[k].shape[1]
      
      pos_list = []
      for i in range(H):
         pos_list.append([])
         for j in range(W):
            pos_list[i].append([i, j, 1])
      X = np.array(pos_list)
      X = X.reshape(-1, 3)

      Y = X @ Fn.transpose()
      Expand = (1 / Y[:, 2]).reshape(-1, 1)
      Y = Y * Expand # 讓第三維變為1
      Y = Y.astype(np.int_) # 取x y座標 忽略第三維
      rows = Y[:, 0]
      cols = Y[:, 1]
      canvas = self.canvas
      canvas[rows, cols] = self.imgs[k].reshape(-1, 3)
      self.canvas = canvas


def main():
   H = Homography()
   H.read_file()
   for i in range(5):
      H.solve_homography(i)
   cv2.imwrite(os.path.join('output', 'part1.png'), H.canvas)

if __name__ == '__main__':
   main()
