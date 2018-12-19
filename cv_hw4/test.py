import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

IMG_LEFT_PATH = sys.argv[1]
IMG_RIGHT_PATH = sys.argv[2]
NAME = sys.argv[3]
MAX_DISP = int(sys.argv[4])
SCALE_FACTOR = int(sys.argv[5])

np.set_printoptions(edgeitems = 5)

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
        img_left_bgr = cv2.imread(l_path)
        img_right_bgr = cv2.imread(r_path)

        self.img_left_bgr = img_left_bgr.astype(np.int_)
        self.img_right_bgr = img_right_bgr.astype(np.int_)

        self.img_left_gray = cv2.cvtColor(img_left_bgr, cv2.COLOR_BGR2GRAY).astype(np.int_)
        self.img_right_gray = cv2.cvtColor(img_right_bgr, cv2.COLOR_BGR2GRAY).astype(np.int_)
        
        self.img_left_lab = cv2.cvtColor(img_left_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.int_)
        self.img_right_lab = cv2.cvtColor(img_right_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.int_)

        self.row, self.col = self.img_left_gray.shape

    def match_costing(self):
        self._preprocess()
        row, col = self.row, self.col
        self.left_disparity_map = np.zeros((row, col), dtype = np.int_)
        
        # LEFT
        for r in range(row):
            for c in range(col):
                source_bits = self.left_bstrs[r, c]
                mask_bits = self.left_mask_bstrs[r, c]
                ham_dis = 1e9 # initialize
                disparity = None
                for d in range(self.max_disp):
                    if c - d >= 0: 
                        target_bits = self.right_bstrs[r, c - d]
                        count = np.count_nonzero(source_bits ^ target_bits & mask_bits)
                        # count = np.count_nonzero(source_bits ^ target_bits)
                        if count < ham_dis:
                            disparity = d
                            ham_dis = count
                self.left_disparity_map[r, c] = disparity * self.scale_factor
            print('matching_cost left', r)
            cv2.imwrite('output/{}-left.png'.format(NAME), self.left_disparity_map)
        # RIGHT
        self.right_disparity_map = np.zeros((row, col), dtype = np.int_)
        for r in range(row):
            for c in range(col):
                source_bits = self.right_bstrs[r, c]
                mask_bits = self.right_mask_bstrs[r, c]
                ham_dis = 1e9 # initialize
                disparity = None
                for d in range(self.max_disp):
                    if c + d < col: 
                        target_bits = self.left_bstrs[r, c + d]
                        count = np.count_nonzero(source_bits ^ target_bits & mask_bits)
                        # count = np.count_nonzero(source_bits ^ target_bits)
                        if count < ham_dis:
                            disparity = d
                            ham_dis = count
                self.right_disparity_map[r, c] = disparity * self.scale_factor
            print('matching_cost right', r)
            cv2.imwrite('output/{}-right.png'.format(NAME), self.right_disparity_map)
        
    def skip_cost(self):
        self.left_disparity_map = cv2.imread('output/{}-left.png'.format(NAME), 0)
        self.right_disparity_map = cv2.imread('output/{}-right.png'.format(NAME), 0)

    def refine_median(self):
        left_disp = self.left_disparity_map.astype(np.uint8)
        kernal_size = 7
        left_disp = cv2.medianBlur(left_disp, kernal_size)
        left_disp = cv2.bilateralFilter(left_disp, 10, 9, 16)
        cv2.imwrite('{}.png'.format(NAME), left_disp)

    def refine_BSM(self):
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

        temp_map = valid_map.copy().astype(np.int_) * 255
        # cv2.imwrite('valid_map.png', temp_map)

        r_axis = np.tile(range(col), (row, 1))
        c_axis = np.tile(range(row), (col, 1)).T

        LDc, LDe = 9, 16
        refine_threshold = self.max_disp // 6
        for r in range(row):
            for c in range(col):
                if valid_map[r, c] == False:
                    disparity = 1e5
                    for k in range(col):
                        if c - k < 0:
                            break
                        elif valid_map[r, c - k] == True and left_disp[r, c - k] > refine_threshold:
                            disparity = min(disparity, left_disp[r, c - k])
                            break
                    for k in range(col):
                        if c + k >= col:
                            break
                        elif valid_map[r, c + k] == True and left_disp[r, c + k] > refine_threshold:
                            disparity = min(disparity, left_disp[r, c + k])
                            break
                    new_disp[r, c] = disparity
            cv2.imwrite('output/{}.png'.format(NAME), new_disp * self.scale_factor)
            print(r)
        self.left_disparity_map = new_disp * self.scale_factor
                    

    def _preprocess(self):
        row, col = self.row, self.col
        left_bstrs = np.zeros((row, col, self.des_len), dtype = np.bool_)
        right_bstrs = np.zeros((row, col, self.des_len), dtype = np.bool_)
        left_mask_bstrs = np.zeros((row, col, self.des_len), dtype = np.bool_)
        right_mask_bstrs = np.zeros((row, col, self.des_len), dtype = np.bool_)

        img_left_pad = np.pad(self.img_left_gray, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))
        img_right_pad = np.pad(self.img_right_gray, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))
        
        img_left_lab_pad = np.pad(self.img_left_lab, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))
        img_right_lab_pad = np.pad(self.img_right_lab, (self.patch_size // 2, self.patch_size // 2), 'constant', constant_values=(0, 0))
        
        for r in range(row):
            for c in range(col):
                left_bstrs[r, c] = self._findbinstring(r, c, img_left_pad)
                right_bstrs[r, c] = self._findbinstring(r, c, img_right_pad)
                left_mask_bstrs[r, c] = self._findmask(r, c, img_left_lab_pad)
                right_mask_bstrs[r, c] = self._findmask(r, c, img_right_lab_pad)
            print('_preprocess', r)
        self.left_bstrs = left_bstrs
        self.right_bstrs = right_bstrs
        self.left_mask_bstrs = left_mask_bstrs
        self.right_mask_bstrs = right_mask_bstrs

    def _findbinstring(self, r_offset, c_offset, image_pad):
        pad = self.patch_size // 2
        p_r = self.p_samples[:, 0] + r_offset + pad
        p_c = self.p_samples[:, 1] + c_offset + pad
        q_r = self.q_samples[:, 0] + r_offset + pad
        q_c = self.q_samples[:, 1] + c_offset + pad
        bits = image_pad[p_r, p_c] > image_pad[q_r, q_c]
        return bits

    def _findmask(self, r_offset, c_offset, image_pad):
        pad = self.patch_size // 2
        p_r = self.p_samples[:, 0] + r_offset + pad
        p_c = self.p_samples[:, 1] + c_offset + pad
        q_r = self.q_samples[:, 0] + r_offset + pad
        q_c = self.q_samples[:, 1] + c_offset + pad
        now_val = image_pad[r_offset + pad, c_offset + pad]
        p_val = image_pad[p_r, p_c]
        q_val = image_pad[q_r, q_c]
        sad_xp = np.absolute(p_val - now_val)
        sad_xq = np.absolute(q_val - now_val)
        w = np.maximum(sad_xp, sad_xq)
        w_sort = np.sort(w)
        quarter_smallest = w_sort[len(w_sort) // 4]
        bits = (w <= quarter_smallest)
        return bits


if __name__ == '__main__':
    a = BSM(MAX_DISP, SCALE_FACTOR)
    a.setPairDistr()
    a.load_image(IMG_LEFT_PATH, IMG_RIGHT_PATH)
    # a.match_costing()
    a.skip_cost()
    a.refine_BSM()
    a.refine_median()