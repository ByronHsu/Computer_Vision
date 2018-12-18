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

def read_file():
   canvas_name = 'times_square.jpg'
   imgs_name = ['wu.jpg', 'ding.jpg', 'yao.jpg', 'kp.jpg', 'lee.jpg']
   canvas_corners = [
      np.array([[352, 818], [407, 818], [352, 884], [408, 885]]),
      np.array([[14, 311], [152, 157], [150, 402], [315, 278]]),
      np.array([[674, 364], [864, 279], [725, 430], [885, 369]]),
      np.array([[495, 808], [609, 802], [495, 892], [609, 896]]),
      np.array([[608, 1024], [664, 1032], [593, 1118], [651, 1134]])
   ]
   canvas = cv2.imread(os.path.join('input', canvas_name))
   imgs = []
   for i in range(5):
      img = cv2.imread(os.path.join('input', imgs_name[i]))
      corner = np.array([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
      imgs.append(img)
   return canvas, imgs, canvas_corners

def solve_homography(u, v):
   A = []
   for i in range(4):
      t = point_to_vec(u[i], v[i])
      A.append(t[0])
      A.append(t[1])
   A = np.array(A)
   u, s, vh = np.linalg.svd(A.transpose() @ A, full_matrices = False)
   Fn = u[:, 8].reshape(3, 3)
   return Fn

def transform(img, canvas, corners):
   img_corners = np.array([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
   canvas_corners = corners
   Fn = solve_homography(img_corners, canvas_corners)
   
   H, W = img.shape[0], img.shape[1]
   pos_list = []
   for i in range(H):
      pos_list.append([])
      for j in range(W):
         pos_list[i].append([i, j, 1])
   X = np.array(pos_list).reshape(-1, 3)
   Y = X @ Fn.transpose()
   Expand = (1 / Y[:, 2]).reshape(-1, 1)
   Y = Y * Expand # 讓第三維變為1
   Y = Y.astype(np.int_)
   # 取x y座標 忽略第三維
   rows = Y[:, 0].reshape(H, W)
   cols = Y[:, 1].reshape(H, W)
   canvas[rows, cols] = img


def time_square():
   canvas, imgs, canvas_corners = read_file()
   for i in range(len(imgs)):
      transform(imgs[i], canvas, canvas_corners[i])
   cv2.imwrite(os.path.join('output', 'part1.png'), canvas)

def transform_inv(img, canvas, canvas_corners, img_corners):
   Fn = solve_homography(img_corners, canvas_corners)
   Fn_i = np.linalg.inv(Fn) # inverse

   H, W = canvas.shape[0], canvas.shape[1]
   pos_list = []
   for i in range(H):
      pos_list.append([])
      for j in range(W):
         pos_list[i].append([i, j, 1])

   X = np.array(pos_list).reshape(-1, 3)
   Y = X @ Fn_i.transpose()
   Expand = (1 / Y[:, 2]).reshape(-1, 1)
   Y = Y * Expand # 讓第三維變為1
   Y = Y.astype(np.int_)

   # 取x y座標 忽略第三維
   rows = Y[:, 0].reshape(H, W)
   cols = Y[:, 1].reshape(H, W)

   canvas[:] = img[rows, cols]

def qr_code():
   canvas = np.zeros((500, 500, 3), dtype = np.int_)
   img = cv2.imread(os.path.join('input', 'screen.jpg'))
   img_corners = [[368, 1040], [550, 977], [400, 1100], [600, 1040]]
   canvas_corners = [[0, 0], [500, 0], [0, 500], [500, 500]]
   transform_inv(img, canvas, canvas_corners, img_corners)
   cv2.imwrite(os.path.join('output', 'part2.png'), canvas)

def top_view():
   canvas = np.zeros((500, 800, 3), dtype = np.int_)
   img = cv2.imread(os.path.join('input', 'crosswalk_front.jpg'))
   img_corners = [[143, 145], [286, 0], [140, 575], [286, 725]]
   canvas_corners = [[0, 0], [500, 0], [0, 800], [500, 800]]
   transform_inv(img, canvas, canvas_corners, img_corners)
   cv2.imwrite(os.path.join('output', 'part3.png'), canvas)

def main():
   # part 1
   time_square()
   # part 2
   qr_code()
   # part 3
   top_view()

if __name__ == '__main__':
   main()
