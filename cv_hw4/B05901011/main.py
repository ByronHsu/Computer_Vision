import numpy as np
import cv2
import time
from model import BSM

def computeDisp(Il, Ir, max_disp, scale_factor):
    model = BSM(max_disp, scale_factor)
    model.setPairDistr()
    model.load_image(Il, Ir)

    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    model.match_costing()
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    model.refine_BSM()
    model.refine_median()
    # ex: Left-right consistency check + hole filling + weighted median filtering
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return model.left_disparity_map


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp, scale_factor)
    cv2.imwrite('tsukuba.png', np.uint8(labels))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp, scale_factor)
    cv2.imwrite('venus.png', np.uint8(labels))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp, scale_factor)
    cv2.imwrite('teddy.png', np.uint8(labels))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp, scale_factor)
    cv2.imwrite('cones.png', np.uint8(labels))

if __name__ == '__main__':
    main()
