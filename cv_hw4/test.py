import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

IMG_LEFT_PATH = sys.argv[1]
IMG_RIGHT_PATH = sys.argv[2]
OUTPUT_PATH = sys.argv[3]
MAX_DISP = int(sys.argv[4])
SCALE_FACTOR = int(sys.argv[5])


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
        self.max_disp = max_disp
        self.scale_factor = scale_factor
    def setPairDistr(self):
        # 4096 x 2
        p_samples = np.zeros((self.des_len, 2), dtype = np.float_)
        p_samples[:, 0] = np.random.normal(0, self.std, self.des_len).T
        p_samples[:, 1] = np.random.normal(0, self.std, self.des_len).T
        p_samples = np.clip(p_samples.astype(np.int_), -13, 13)
        
        q_samples = np.zeros((self.des_len, 2), dtype = np.float_)
        q_samples[:, 0] = np.random.normal(0, self.std, self.des_len).T
        q_samples[:, 1] = np.random.normal(0, self.std, self.des_len).T
        q_samples = np.clip(q_samples.astype(np.int_), -13, 13)

        self.p_samples = p_samples
        self.q_samples = q_samples
        print('finish setting pair distribution')

    def load_image(self, l_path, r_path):
        self.img_left = cv2.imread(l_path, 0)
        self.img_right = cv2.imread(r_path, 0)
        self.row, self.col = self.img_left.shape
    def match(self):
        self._preprocess()
        row, col = self.row, self.col
        disparity_map = np.zeros((row, col), dtype = np.int_)
        for r in range(row):
            for c in range(col):
                source_bits = self.left_bstrs[r, c]
                ham_dis = 1e9 # initialize
                disparity = None
                for d in range(self.max_disp):
                    if c - d >= 0: 
                        target_bits = self.right_bstrs[r, c - d]
                        count = np.count_nonzero(source_bits ^ target_bits)
                        if count < ham_dis:
                            disparity = d
                            ham_dis = count
                disparity_map[r, c] = disparity * self.scale_factor
            cv2.imwrite(OUTPUT_PATH, disparity_map)
            print('match', r)

        cv2.imwrite(OUTPUT_PATH, disparity_map)
        print('writing {}'.format(OUTPUT_PATH))
    def _preprocess(self):

        row, col = self.row, self.col
        left_bstrs = np.zeros((row, col, self.des_len), dtype = np.bool_)
        right_bstrs = np.zeros((row, col, self.des_len), dtype = np.bool_)

        img_left_pad = np.pad(self.img_left, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))
        img_right_pad = np.pad(self.img_right, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))

        for r in range(row):
            for c in range(col):
                left_bstrs[r, c] = self._findbinstring(r, c, img_left_pad)
                right_bstrs[r, c] = self._findbinstring(r, c, img_right_pad)
            print('_preprocess', r)
        self.left_bstrs = left_bstrs
        self.right_bstrs = right_bstrs

    def _findbinstring(self, r_offset, c_offset, image_pad):
        pad = self.patch_size // 2
        p_r = self.p_samples[:, 0] + r_offset + pad
        p_c = self.p_samples[:, 1] + c_offset + pad
        q_r = self.q_samples[:, 0] + r_offset + pad
        q_c = self.q_samples[:, 1] + c_offset + pad
        bits = image_pad[p_r, p_c] > image_pad[q_r, q_c]
        return bits

if __name__ == '__main__':
    a = BSM(MAX_DISP, SCALE_FACTOR)
    a.setPairDistr()
    a.load_image(IMG_LEFT_PATH, IMG_RIGHT_PATH)
    a.match()
