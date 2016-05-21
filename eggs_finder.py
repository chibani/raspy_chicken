#! /usr/local/bin/python

import os
import cv2
import numpy as np;
from datetime import date
import re
import csv

import json

with open('config.json', 'r') as f:
    config = json.load(f)

def detect_eggs(im, params):
    detector = cv2.SimpleBlobDetector(params)
    # Detect eggs (blobs)
    keypoints = detector.detect(im)
    return keypoints

def extract_should_have(filename):
    reg = re.compile("^(\d+)-")
    return reg.match(filename).group(1)

def find_eggs_in_directory(originals_path, params):
    total_images = len([name for name in os.listdir(originals_path) if os.path.isfile(originals_path+name)])
    accurate_detections = 0

    for fn in os.listdir(originals_path):
        if os.path.isfile(originals_path+fn):
            detection_result = find_eggs_in_file(originals_path+fn, params)
            expected_number = int(extract_should_have(fn))

            im = detection_result[0]
            keypoints = detection_result[1]

            print "Found %d eggs for %d in %s" % (len(keypoints), expected_number, fn)

            if len(keypoints) == expected_number :
                accurate_detections += 1

            if len(keypoints):


                # Draw detected blobs as red circles.
                im_with_keypoints = mark_image_with_keypoints(im, keypoints)

                result_filename = fn+date.today().isoformat()+".jpg"
                result_filepath = config['photos_dir']+result_filename
                cv2.imwrite(result_filepath, im_with_keypoints)

    print "Accuracy (%d / %d) %f" % (accurate_detections, total_images, (accurate_detections/float(total_images)) )

    with open('bench.csv', 'a') as csvfile:
        params['accuracy'] = accurate_detections/float(total_images)
        fieldnames = ['accuracy','minThreshold','maxThreshold','filterByColor','blobColor','filterByArea','minArea', 'maxArea','filterByCircularity','minCircularity', 'maxCircularity','filterByConvexity','minConvexity', 'maxConvexity','filterByInertia','minInertiaRatio','maxInertiaRatio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writer.writeheader()
        writer.writerow(params)


def find_eggs_in_file(filepath, params):
    # Read image
    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    # im = cv2.imread(originals_path+fn, cv2.CV_LOAD_IMAGE_COLOR)

    keypoints = detect_eggs(im, get_blob_detector_params(params) )
    return (im,keypoints)

def mark_image_with_keypoints(im, keypoints):
    return cv2.drawKeypoints(im, keypoints, np.array([]), (10,10,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def get_blob_detector_params(my_params):
    # create a SimpleBlobDetector param container
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = my_params.get('minThreshold')
    params.maxThreshold = my_params.get('maxThreshold')

    # Color
    params.filterByColor = my_params.get('filterByColor')
    params.blobColor = my_params.get('blobColor')

    # Filter by Area.
    params.filterByArea = my_params.get('filterByArea')
    params.minArea = my_params.get('minArea')
    params.maxArea = my_params.get('maxArea')


    # Filter by Circularity
    params.filterByCircularity = my_params.get('filterByCircularity')
    params.minCircularity = my_params.get('minCircularity')
    params.maxCircularity = my_params.get('maxCircularity')

    # Filter by Convexity
    params.filterByConvexity = my_params.get('filterByConvexity')
    params.minConvexity = my_params.get('minConvexity')
    params.maxConvexity = my_params.get('maxConvexity')

    # Filter by Inertia
    params.filterByInertia = my_params.get('filterByInertia')
    params.minInertiaRatio = my_params.get('minInertiaRatio')
    params.maxInertiaRatio = my_params.get('maxInertiaRatio')

    return params



if __name__ == '__main__':
    params = {}
    params['minThreshold'] = 80
    params['maxThreshold'] = 250

    # Color
    params['filterByColor'] = True
    params['blobColor'] = 255

    # Filter by Area.
    params['filterByArea'] = True
    params['minArea'] = 500
    params['maxArea'] = 3000

    # Filter by Circularity
    params['filterByCircularity'] = True
    params['minCircularity'] = 0.3
    params['maxCircularity'] = 0.9

    # Filter by Convexity
    params['filterByConvexity'] = True
    params['minConvexity'] = 0.4
    params['maxConvexity'] = 1

    # Filter by Inertia
    params['filterByInertia'] = True
    params['minInertiaRatio'] = 0.3
    params['maxInertiaRatio'] = 0.6

    find_eggs_in_directory( './originals/', params )

    #result = find_eggs_in_file('./originals/3-IMG_20160309_082415.jpg', params )
    #print "%d eggs found" % len(result[1])
