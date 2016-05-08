#! /usr/local/bin/python

import os
import cv2
import numpy as np;
from datetime import date

import json

with open('config.json', 'r') as f:
    config = json.load(f)

def detect_eggs(im):
    # Set up the detector
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 80;#150
    params.maxThreshold = 250; #250

    # Color
    params.filterByColor = True
    params.blobColor = 255 #255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 2500 #900

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3 #0.2

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.7 #0.7

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 0.7

    detector = cv2.SimpleBlobDetector(params)
    # Detect eggs (blobs)
    keypoints = detector.detect(im)
    return keypoints

originals_path = './originals/'
for fn in os.listdir(originals_path):
    if os.path.isfile(originals_path+fn):
        # Read image
        im = cv2.imread(originals_path+fn, cv2.IMREAD_GRAYSCALE)
        keypoints = detect_eggs(im)
        print "Found %d eggs " % len(keypoints)

        if len(keypoints) > 0:
            # Draw detected blobs as red circles.
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (10,10,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            #notify_for_eggs(len(keypoints), im_with_keypoints)
            # Show keypoints
            #cv2.imshow("%d egg(s)" % len(keypoints), im_with_keypoints)
            #cv2.waitKey(0)
            result_filename = fn+date.today().isoformat()+".jpg"
            result_filepath = config['photos_dir']+result_filename
            cv2.imwrite(result_filepath, im_with_keypoints)
