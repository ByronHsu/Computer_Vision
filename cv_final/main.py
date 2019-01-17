import numpy as np
import argparse
import cv2
import time
from util import writePFM
import model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Disparity Estimation')
parser.add_argument('--input-left', default='./data/Synthetic/TL0.png', type=str, help='input left image')
parser.add_argument('--input-right', default='./data/Synthetic/TR0.png', type=str, help='input right image')
parser.add_argument('--output', default='./TL0.pfm', type=str, help='left disparity map')


# You can modify the function interface as you like
def computeDisp(l_path, r_path):
    model.evaluate(l_path, r_path)


def main():
    args = parser.parse_args()
    print('Compute disparity for %s' % args.input_left)
    img_left = cv2.imread(args.input_left)
    img_right = cv2.imread(args.input_right)
    tic = time.time()
    # disp = computeDisp(img_left, img_right)
    disp = computeDisp(args.input_left, args.input_right)
    toc = time.time()
    writePFM(args.output, disp)
    plt.imsave(args.output.replace('.pfm', '.png'), disp, cmap = 'jet')
    
    print('Elapsed time: %f sec.' % (toc - tic))


if __name__ == '__main__':
    main()
