#!/usr/bin/env python3

################################################################################
# __        ___           _
# \ \      / / |__   __ _| |_ ___ _   _ _ __
#  \ \ /\ / /| '_ \ / _` | __/ __| | | | '_ \
#   \ V  V / | | | | (_| | |_\__ \ |_| | |_) |
#    \_/\_/  |_| |_|\__,_|\__|___/\__,_| .__/
#                                      |_|
#
# 2020-04-30: An updated version of Stuporglue's “Whatsup” script. Now uses
# Python3 and CV2 methods and conventions.
#
# Usage: whatsup [filepath]
#
# Output: The number of degrees it should be rotated clockwise to orient the faces correctly.
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
# Primary source:
#     https://stuporglue.org/automatically-orient-scanned-photos-correctly-with-opencv/
#
# Two other — now dead — sites are referenced as a source as well:
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
    'FULLBODY': 'haarcascade_fullbody.xml',
    'HAAR_FRONTALFACE_ALT_TREE': 'haarcascade_frontalface_alt_tree.xml',
    'HAAR_FRONTALFACE_DEFAULT': 'haarcascade_frontalface_default.xml',
    'HAAR_FRONTALFACE_ALT': 'haarcascade_frontalface_alt.xml',
    'HAAR_FRONTALFACE_ALT2': 'haarcascade_frontalface_alt2.xml',
}

################################################################################
# The 'manage_face_detection' function.
def manage_face_detection(biggest=False):

    ############################################################################
    # Set the filename from the input argument.
    filename_full = sys.argv[-1]

    ############################################################################
    # Set the filename and extension.
    filename = pathlib.Path(filename_full).stem
    extension = pathlib.Path(filename_full).suffix

    ############################################################################
    # Set the image path.
    image_path = os.path.abspath(filename_full)

    ############################################################################
    # Load the image into the script.
    image = cv2.imread(image_path)

    ############################################################################
    # Adjust contrast and brightness: Contrast (1.0-3.0), Brightness (0-100)
    contrast = 1.25
    brightness = 0
    image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    ############################################################################
    # Convert the image to grayscale.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ############################################################################
    # Equalize the histogram.
    image = cv2.equalizeHist(image)

    ############################################################################
    # Roll through the cascades.
    for key, THIS_CASCADE in PROFILES.items():

        ########################################################################
        # Initialize the counter.
        counter = 2
        count_minimum = 1

        ########################################################################
        # Define the cascade classifier.
        cc = cv2.CascadeClassifier(os.path.join(DATA_DIRECTORY, THIS_CASCADE))

        ########################################################################
        # Roll through the sizes.
        while counter >= count_minimum:

            ####################################################################
            # Get the dimensions of the image.
            image_h, image_w = image.shape[:2]

            ####################################################################
            # Calculate the new size for the images.
            resize_h = round(image_h / counter)
            resize_w = round(image_w / counter)

            ####################################################################
            # Resize the image.
            image_resized = cv2.resize(image, (resize_w, resize_h), interpolation = cv2.INTER_CUBIC)

            ####################################################################
            # Send the image to the 'face_detection' method.
            results = face_detection(image_resized, cc, filename, extension, biggest)

            ####################################################################
            # If we have results return the results.
            if results is not False:
                return results

            counter = counter - 1

    ############################################################################
    # If no faces are found, use the brightest side for orientation instead.
    return bright_side_detection(image, filename, extension)

################################################################################
# The 'face_detection' function.
def face_detection(image, cc, filename, extension, biggest=False):

    ############################################################################
    # Initialize the counter.
    counter = 0
    rotation_maximum = 4

    ############################################################################
    # Set the min and max image size.
    side = math.sqrt(image.size)
    min_length = int(side / 20)
    max_length = int(side / 2)

    ############################################################################
    # Set the CV2 flags.
    flags = cv2.CASCADE_DO_CANNY_PRUNING

    ############################################################################
    # If we are looking for the biggest face, set that flag.
    if biggest:
        flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT

    ############################################################################
    # Roll through the rotations to use.
    while counter < rotation_maximum:


        # ########################################################################
        # # deal with the gray image
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im = cv2.equalizeHist(im)

        # ########################################################################
        # # frontal faces
        # cc1 = CASCADES['HAAR_FRONTALFACE_ALT2']
        # cc2 = CASCADES['HAAR_FRONTALFACE_DEFAULT']
        # faces_found = cc1.detectMultiScale(im, 1.3, 6, flags, (minlen, minlen), (maxlen, maxlen))
        # if len(features) == 0:
        #     faces_found = cc2.detectMultiScale(im, 1.4, 6, flags, (minlen, minlen), (maxlen, maxlen))

        ########################################################################
        # Let's attempt to detect some faces.
        faces_found = cc.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))

        ########################################################################
        # TODO: Debugging stuff.
        if debug:
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
        # If a face is found, multiply the counter by 90 to get the number of degrees the image should be rotated.
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

        ########################################################################
        # Rotate the image 90 degrees clockwise.
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        ########################################################################
        # Increment the counter.
        counter = counter + 1

    return False

################################################################################
# The 'bright_side_detection' function.
def bright_side_detection(image, filename, extension):

    ############################################################################
    # Set the ratio used to slice up the image.
    ratio = 3
    boundary = (ratio - 1)

    ############################################################################
    # Set the tuple for resize dimensions.
    resize = (5, 5)

    ############################################################################
    # Set the tuple for kernel size.
    blur_kernel = (5, 5)

    ############################################################################
    # Set the mapping for rotation values.
    rotation_map = { 'top': 0, 'left': 90, 'bottom': 180, 'right': 270 }

    ############################################################################
    # Get the dimensions of the image.
    (image_h, image_w) = image.shape[:2]

    ############################################################################
    # Get sample chunks.
    chunks = {}
    chunks['top'] = image[0:round(image_h / ratio), 0:image_w]
    chunks['left'] = image[0:image_h, 0:round(image_w / ratio)]
    chunks['bottom'] = image[round(boundary * (image_h / ratio)):image_h, 0:image_w]
    chunks['right'] = image[0:image_h, round(boundary * (image_w / ratio)):image_w]

    ####################################################################
    # Resize and blur the images to average things out.
    samples = {}
    for position in chunks:
        samples[position] = cv2.mean(cv2.GaussianBlur(cv2.resize(chunks[position], resize, interpolation = cv2.INTER_CUBIC), blur_kernel, cv2.BORDER_DEFAULT))[0]

    ############################################################################
    # Get the max value from the samples.
    max_side = max(samples, key = samples.get)

    ############################################################################
    # Return the final return value.
    return rotation_map[max_side]

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
# print (rotation)
print (image_data_string)

################################################################################
# TODO: Some simple debugging. Don't use Python to do image writing.
# Instead use the output with a batch processor like ImageMagick.
if debug:
    filename = pathlib.Path(sys.argv[-1]).stem
    extension = pathlib.Path(sys.argv[-1]).suffix
    image_path = os.path.abspath(sys.argv[-1])

    image = cv2.imread(image_path)
    image = rotate_image(image, rotation)

    image_test = filename + '_' + str(rotation) + extension
    image_data_string = ' ' . join(str(value) for value in image_data.values())

    cv2.imwrite(image_test, image)
