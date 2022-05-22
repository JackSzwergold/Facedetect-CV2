#!/usr/bin/env python3

################################################################################
#  _____                ____       _            _     ____   ___ ____  ____
# |  ___|_ _  ___ ___  |  _ \  ___| |_ ___  ___| |_  |___ \ / _ \___ \|___ \
# | |_ / _` |/ __/ _ \ | | | |/ _ \ __/ _ \/ __| __|   __) | | | |__) | __) |
# |  _| (_| | (_|  __/ | |_| |  __/ ||  __/ (__| |_   / __/| |_| / __/ / __/
# |_|  \__,_|\___\___| |____/ \___|\__\___|\___|\__| |_____|\___/_____|_____|
#
# 2022-05-31: An updated version of the Face Detect script that also detects
# image orientation.
#
# Usage: facedetect2022 [filepath]
#
# Output: The X and Y coordinates as well as the width and height of the detected
# face as well as the number of degrees it should be rotated clockwise to orient
# the faces correctly.
#
################################################################################

################################################################################
#
# As the author describes it:
#
# “This script reads in a file and counter to determine which orientation is
# correct by looking for faces in the photos. It starts with the existing
# orientation, then rotates it 90 degrees at a time until it has either tried
# all 4 directions or until it finds a face”
#
# Primary source that is now dead:
#     https://stuporglue.org/automatically-orient-scanned-photos-correctly-with-opencv/
#
# Another two other, now dead, sites are referenced as a source as well:
#     http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
#     http://opencv.willowgarage.com/documentation/python/core_operations_on_arrays.html#createmat
#
################################################################################

################################################################################
# Import various modules and functions.
import sys
import os
import cv2
import math
import numpy as np;
import pathlib

################################################################################
# Enable debug mode.
debug = True

################################################################################
# Set the cascade data directory, cascades and profiles.
DATA_DIRECTORY = cv2.data.haarcascades
CASCADES = {}
PROFILES = {
    'HAAR_PROFILEFACE': 'haarcascade_profileface.xml',
    'HAAR_FULLBODY': 'haarcascade_fullbody.xml',
    'HAAR_FRONTALFACE_DEFAULT': 'haarcascade_frontalface_default.xml',
    'HAAR_FRONTALFACE_ALT2': 'haarcascade_frontalface_alt2.xml',
    'HAAR_FRONTALFACE_ALT': 'haarcascade_frontalface_alt.xml',
}

################################################################################
# The 'manage_face_detection' function.
def manage_face_detection(biggest= False):

    ############################################################################
    # Set the defaults to return if actual face detection is false.
    default = {
        'x': 0,
        'y': 0,
        'w': 0,
        'h': 0,
        'd': 0,
    }

    ############################################################################
    # Set the filename from the input argument.
    filename_full = sys.argv[-1]

    ############################################################################
    # Set the filename and extension.
    filename = pathlib.Path(filename_full).stem
    extension = pathlib.Path(filename_full).suffix

    ############################################################################
    # Set the image path.
    image_filepath = os.path.abspath(filename_full)

    ############################################################################
    # Load the image into the script.
    image = cv2.imread(image_filepath)

    ############################################################################
    # Send the image to the 'face_detection' method.
    results = face_detection(image, filename, extension, False)

    ############################################################################
    # If we have results, then return the results.
    if results is not False:
        return results
    else:
        return default

################################################################################
# The 'face_detection' function.
def face_detection(image_source, filename, extension, biggest=False):

    ############################################################################
    # Initialize the counter stuff.
    counter = 0
    rotation_max = 4

    ###########################################################################
    # Adjust the image for face detection purposes.
    contrast = 2.50
    brightness = 0
    image = cv2.convertScaleAbs(image_source, alpha = contrast, beta = brightness)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    ############################################################################
    # Set the CV2 flags.
    flags = cv2.CASCADE_DO_CANNY_PRUNING

    ############################################################################
    # If we are looking for the biggest face, set that flag.
    if biggest:
        flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT

    ############################################################################
    # Set the cascades.
    cc_alt = CASCADES['HAAR_FRONTALFACE_ALT']
    cc_alt2 = CASCADES['HAAR_FRONTALFACE_ALT2']
    cc_default = CASCADES['HAAR_FRONTALFACE_DEFAULT']
    cc_fullbody = CASCADES['HAAR_FULLBODY']
    cc_profileface = CASCADES['HAAR_PROFILEFACE']

    ############################################################################
    # Roll through the rotations to use.
    while counter < rotation_max:

        ########################################################################
        # Set the min and max image size.
        side = math.sqrt(image.size)
        min_length = int(side / 20)
        max_length = int(side / 2)

        ########################################################################
        # Try and find faces.
        faces_found = cc_alt2.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = cc_default.detectMultiScale(image, 1.4, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = cc_default.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = cc_alt.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = cc_fullbody.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = cc_profileface.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))

        ########################################################################
        # TODO: Some simple debugging. Don't use Python to do image writing.
        # Instead use the output with a batch processor like ImageMagick.
        if debug:
            image_test = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image_test_filename = filename + '_test' + extension
            cv2.imwrite(image_test_filename, image_test)
            for x, y, w, h in faces_found:
                start_point = (x, y)
                end_point = (x + w, y + h)
                color = (0, 255, 0)
                thickness = 5
                image_facebox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                image_facebox = cv2.rectangle(image_facebox, start_point, end_point, color, thickness)
                image_facebox_filename = filename + '_facebox' + extension
                cv2.imwrite(image_facebox_filename, image_facebox)

        ########################################################################
        # If a face is found, return the values. Or else, multiply the counter
        # by 90 to get the number of degrees the image should be rotated.
        if (len(faces_found) > 0):
            rotation = counter * 90
            final = {
                'x': int(faces_found[0][0]),
                'y': int(faces_found[0][1]),
                'w': int(faces_found[0][2]),
                'h': int(faces_found[0][3]),
                'd': int(rotation),
            }
            return final
            break
        else:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            counter = counter + 1

    return False

################################################################################
# The 'rotate_image' function.
# Source: https://stackoverflow.com/a/58127701/117259
def rotate_image(image, angle):

    ############################################################################
    # Grab the dimensions of the image and then determine the center
    (image_h, image_w) = image.shape[:2]
    (cX, cY) = (image_w / 2, image_h / 2)

    ############################################################################
    # Grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    ############################################################################
    # Compute the new bounding dimensions of the image
    nW = int((image_h * sin) + (image_w * cos))
    nH = int((image_h * cos) + (image_w * sin))

    ############################################################################
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    ############################################################################
    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

################################################################################
# The 'fatal' function.
def fatal(msg):
    error(msg)
    sys.exit(1)

################################################################################
# The 'load_cascades' function.
def load_cascades(data_dir):
    for k, v in PROFILES.items():
        v = os.path.join(data_dir, v)
        try:
            if not os.path.exists(v):
                raise cv2.error('no such file')
            CASCADES[k] = cv2.CascadeClassifier(v)
        except cv2.error:
            fatal("cannot load {} from {}".format(k, v))

################################################################################
# Usage Check
if (len(sys.argv) != 2):
    print ("USAGE: whatsup filename")
    sys.exit(-1)

################################################################################
# Sanity check
if not os.path.isfile(sys.argv[-1]):
    print ("File '" + sys.argv[-1] + "' not found.")
    sys.exit(-1)

################################################################################
# And here's where we invoke it and get the the output.
load_cascades(DATA_DIRECTORY)

################################################################################
# And here's where we invoke it and get the the output.
image_data = manage_face_detection(True)

################################################################################
# Get the rotation from the image data.
rotation = int(image_data['d'])

################################################################################
# Set the image data string.
image_data_string = ' ' . join(str(value) for value in image_data.values())

################################################################################
# Return the final return value.
print (image_data_string)

################################################################################
# TODO: Some simple debugging. Don't use Python to do image writing.
# Instead use the output with a batch processor like ImageMagick.
if debug:
    filename = pathlib.Path(sys.argv[-1]).stem
    extension = pathlib.Path(sys.argv[-1]).suffix
    image_filepath = os.path.abspath(sys.argv[-1])
    image = cv2.imread(image_filepath)
    image = rotate_image(image, rotation)
    image_test = filename + '_' + str(rotation) + extension
    image_data_string = ' ' . join(str(value) for value in image_data.values())
    cv2.imwrite(image_test, image)
