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

    def match_costing(self):
        self._preprocess()
        row, col = self.row, self.col
        self.left_disparity_map = np.zeros((row, col), dtype = np.int_)
        
        # LEFT
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
                self.left_disparity_map[r, c] = disparity * self.scale_factor
            print('matching_cost left', r)

            cv2.imwrite('output/cones-left.png', self.left_disparity_map)

        # RIGHT
        self.right_disparity_map = np.zeros((row, col), dtype = np.int_)
        for r in range(row):
            for c in range(col):
                source_bits = self.right_bstrs[r, c]
                ham_dis = 1e9 # initialize
                disparity = None
                for d in range(self.max_disp):
                    if c + d < col: 
                        target_bits = self.left_bstrs[r, c + d]
                        count = np.count_nonzero(source_bits ^ target_bits)
                        if count < ham_dis:
                            disparity = d
                            ham_dis = count
                self.right_disparity_map[r, c] = disparity * self.scale_factor
            print('matching_cost right', r)

        cv2.imwrite('output/cones-right.png', self.right_disparity_map)

    def skip_cost(self):
        
        self.left_disparity_map = cv2.imread('output/cones-left.png', 0)
        self.right_disparity_map = cv2.imread('output/cones-right.png', 0)

    def refine(self):
        row, col = self.row, self.col
        valid_map = np.zeros((row, col), dtype = np.bool_)
        left_disp = self.left_disparity_map.astype(np.int_) // self.scale_factor
        right_disp = self.right_disparity_map.astype(np.int_) // self.scale_factor

        new_disp = left_disp.copy()
        for r in range(row):
            for c in range(col):
                match_r, match_c = (r, c - left_disp[r, c])        
                if match_c < 0:
                    valid_map[r, c] = False
                else:
                    r_disp = right_disp[match_r, match_c]
                    l_disp = left_disp[r, c]
                    if abs(r_disp - l_disp) > 1:
                        valid_map[r, c] = False
                    else:
                        valid_map[r, c] = True

        r_axis = np.tile(range(col), (row, 1))
        c_axis = np.tile(range(row), (col, 1)).T

        LDc, LDe = 9, 16

        for r in range(row):
            for c in range(col):
                if valid_map[r, c] == False:
                    weight_max = 1e-32 # initialize
                    disparity = None
                    for d in range(self.max_disp):
                        index = (valid_map == True) & (left_disp == d)
                        r_fit = r_axis[index]
                        c_fit = c_axis[index]
                        img_fit = self.img_left[index]
                        space_dis = np.sqrt ((r_fit - r) ** 2 + (c_fit - c) ** 2)
                        color_dis = np.sqrt ((img_fit - self.img_left[r, c]) ** 2)
                        exp_index = - space_dis / LDc - color_dis / LDe
                        weight_arr = np.exp(exp_index)
                        weight_sum = np.sum(weight_arr)
                        print(d, r_fit.shape)
                        # print(exp_index)
                        # print(weight_arr)
                        print(weight_sum)
                        # input()
                        if weight_sum > weight_max:
                            disparity = d
                            weight_max = weight_sum 
                    new_disp[r, c] = disparity
            cv2.imwrite('refine.png', new_disp * self.scale_factor)
            print(r)
                    

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
    # a.match_costing()
    a.skip_cost()
    a.refine()