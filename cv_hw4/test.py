import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

IMG_LEFT_PATH = sys.argv[1]
IMG_RIGHT_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
MAX_DISP = int(sys.argv[4])
SCALE_FACTOR = int(sys.argv[5])

img_left = cv2.imread(IMG_LEFT_PATH, 0)
img_right = cv2.imread(IMG_RIGHT_PATH, 0)

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(img_left, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img_right, cmap='gray')
# plt.show()

class BSM():
    def __init__(self, max_disp, scale_factor):
        self.patch_size = 26
        self.des_len = 4096
        self.std = 4
        self.max_disp = 60
        self.scale_factor = 4
    def setPairDistr(self):
        # 4096 x 2
        p_samples = np.zeros((self.des_len, 2), dtype = np.float_)
        p_samples[:, 0] = np.random.normal(0, self.std, self.des_len).T
        p_samples[:, 1] = np.random.normal(0, self.std, self.des_len).T
        self.p_samples = p_samples

        q_samples = np.zeros((self.des_len, 2), dtype = np.float_)
        q_samples[:, 0] = np.random.normal(0, self.std, self.des_len).T
        q_samples[:, 1] = np.random.normal(0, self.std, self.des_len).T
        self.q_samples = q_samples

        self.p_samples = np.clip(self.p_samples.astype(np.int_), -13, 13)
        self.q_samples = np.clip(self.q_samples.astype(np.int_), -13, 13)
        print('finish setting pair distribution')
    def match(self, i1, i2):
        self.img_left = i1
        self.img_right = i2
        row, col = img_left.shape
        disparity_map = np.zeros((row, col), dtype = np.int_)
        for r in range(row):
            for c in range(col):
                source_bits = self._findbinstring(r, c, self.img_left)
                ham_dis = 1e9 # initialize
                disparity = None
                for d in range(self.max_disp):
                    if c - d >= 0: 
                        target_bits = self._findbinstring(r, c - d, self.img_right)
                        count = np.count_nonzero(source_bits ^ target_bits)
                        if count < ham_dis:
                            disparity = d
                            ham_dis = count
                print(r, c, disparity)
                disparity_map[r, c] = disparity * self.scale_factor
                cv2.imwrite(OUTPUT_PATH, disparity_map)
        print(disparity_map)

    def _findbinstring(self, r_offset, c_offset, img):
        pad = self.patch_size // 2
        image_pad = np.pad(img, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))

        p_r = self.p_samples[:, 0] + r_offset + pad
        p_c = self.p_samples[:, 1] + c_offset + pad
        q_r = self.q_samples[:, 0] + r_offset + pad
        q_c = self.q_samples[:, 1] + c_offset + pad
        bits = image_pad[p_r, p_c] > image_pad[q_r, q_c]
        return bits

if __name__ == '__main__':
    a = BSM(MAX_DISP, SCALE_FACTOR)
    a.setPairDistr()
    a.match(img_left, img_right)