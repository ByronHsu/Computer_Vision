import os
import cv2
import numpy as np
from main import transform
from progress.bar import Bar

def detect_and_fill(img_scene, img_object, img_source):
   #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
   minHessian = 400
   detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
   keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
   keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
   
   #-- Step 2: Matching descriptor vectors with a FLANN based matcher
   
   # Since SURF is a floating-point descriptor NORM_L2 is used
   matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
   knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
   
   #-- Filter matches using the Lowe's ratio test
   ratio_thresh = 0.75
   good_matches = []
   for m,n in knn_matches:
      if m.distance < ratio_thresh * n.distance:
         good_matches.append(m)

   #-- Localize the object
   obj = np.empty((len(good_matches),2), dtype=np.float32)
   scene = np.empty((len(good_matches),2), dtype=np.float32)
   for i in range(len(good_matches)):
      #-- Get the keypoints from the good matches
      obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[1] # row
      obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[0] # cols
      scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[1]
      scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[0]

   H, _ =  cv2.findHomography(obj, scene, cv2.RANSAC)

   #-- Get the corners from the image_1 ( the object to be "detected" )
   obj_corners = np.array([[0, 0], [img_object.shape[0], 0], [0, img_object.shape[1]], [img_object.shape[0], img_object.shape[1]]], dtype = np.float_)
   scene_corners = cv2.perspectiveTransform(obj_corners.reshape(4, 1, 2), H)

   #-- Show detected matches
   scene_corners = scene_corners.reshape(4, 2)
   transform(img_source, img_scene, scene_corners)



cap = cv2.VideoCapture(os.path.join('bonus_input', 'ar_marker.mp4'))
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
W, H = 1280, 720

out = cv2.VideoWriter(os.path.join('output', 'bonus.mp4'), fourcc, 20.0, (W, H)) # origin 3840 2160
img_object = cv2.imread(os.path.join('bonus_input', 'marker.png'))
img_object = img_object[40:370, 40:370]
img_source = cv2.imread(os.path.join('bonus_input', 'source.jpg'))


def ar_marker():
   total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   bar = Bar('Processing', max = total_frame)
   while(cap.isOpened()):
      bar.next()
      ret, frame = cap.read()
      if ret == True:
         # print(frame.shape)
         # print(frame.shape)
         # input()
         frame = cv2.resize(frame, (W, H))
         detect_and_fill(frame, img_object, img_source)
         out.write(frame)
      else:
         break
   # Release everything if job is finished
   bar.finish()
   cap.release()
   out.release()

if __name__ == '__main__':
   ar_marker()