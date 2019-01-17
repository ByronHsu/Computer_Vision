from . import classifier
from . import psm
from . import mono
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def classify(img):
    if img.shape[0] == 512:
        return 1
    else:
        return 0

def evaluate(l_path, r_path):
    img_l_gray, img_r_gray = cv2.imread(l_path, 0), cv2.imread(r_path, 0)
    # t = classifier.evaluate(img_l_gray)
    t = classify(img_l_gray)
    disp = None
    img_l, img_r = None, None 
    if t < 0.5:
        img_l, img_r = cv2.imread(l_path), cv2.imread(r_path)
        print('Classified as Real Image...')
        disp = mono.evaluate(img_l)
    else:
        img_l, img_r = Image.open(l_path).convert('RGB'), Image.open(r_path).convert('RGB')
        print('Classified as Syn Image...')
        disp = psm.evaluate(img_l, img_r)

    print('output.shape', disp.shape)
    return disp
