from . import classifier
from . import psm
from . import mono
import cv2
import matplotlib.pyplot as plt

def classify(img):
    if img.shape[0] == 512:
        return 1
    else:
        return 0

def evaluate(l_path, r_path):
    img_l_gray, img_r_gray = cv2.imread(l_path, 0), cv2.imread(r_path, 0)
    # t = classifier.evaluate(img_l_gray)
    t = classify(img_l_gray)
    img_l, img_r = cv2.imread(l_path), cv2.imread(r_path)
    disp = None
    print('input.shape', img_l.shape)
    
    if t < 0.5:
        print('Classified as Real Image...')
        disp = mono.evaluate(img_l)
    else:
        print('Classified as Syn Image...')
        disp = psm.evaluate(img_l, img_r)

    print('output.shape', img_r.shape)
    return disp
