from . import classifier
import cv2

def evaluate(l_path, r_path):
    img_l_gray, img_r_gray = cv2.imread(l_path, 0), cv2.imread(r_path, 0)
    t = classifier.evaluate(img_l_gray)
    print(t)
