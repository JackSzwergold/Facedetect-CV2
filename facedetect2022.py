#!/usr/bin/env python3

################################################################################
#  _____                ____       _            _     ____   ___ ____  ____
# |  ___|_ _  ___ ___  |  _ \  ___| |_ ___  ___| |_  |___ \ / _ \___ \|___ \
# | |_ / _` |/ __/ _ \ | | | |/ _ \ __/ _ \/ __| __|   __) | | | |__) | __) |
# |  _| (_| | (_|  __/ | |_| |  __/ ||  __/ (__| |_   / __/| |_| / __/ / __/
# |_|  \__,_|\___\___| |____/ \___|\__\___|\___|\__| |_____|\___/_____|_____|
#
# 2022-05-21: An updated version of the Face Detect script that also detects
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
import argparse

################################################################################
# Enable debug mode.
debug = True

################################################################################
# Set the cascade data directory, cascades and profiles.
DATA_DIRECTORY = cv2.data.haarcascades
CASCADES_LOADED = {}
CASCADES_DICTIONARY = {
    'ALT2': 'haarcascade_frontalface_alt2.xml',
    'DEFAULT': 'haarcascade_frontalface_default.xml',
}

################################################################################
# The 'manage_face_detection' function.
def manage_face_detection(filename_full, biggest = False):

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
    # Set an array of contrast values.
    contrast_values = {
        1.25,
        2.5,
    }

    ############################################################################
    # Set the filename and extension.
    filename = pathlib.Path(filename_full).stem
    extension = pathlib.Path(filename_full).suffix

    ############################################################################
    # Set the image path.
    image_filepath = os.path.abspath(filename_full)

    ############################################################################
    # Load the image into the script.
    image_source = cv2.imread(image_filepath)

    ############################################################################
    # Roll through the contrast values, and try to detect a face.
    for contrast in contrast_values:
        resize_height = 1800
        blur_factor = 0
        brightness = int(round(255 * (1 - contrast) / 2))

        ########################################################################
        # Crop images for better face detection.
        crop_factor = 20
        crop_x = 0
        crop_y = 0
        if (crop_factor > 0):
            crop_x = int(image_source.shape[1] / crop_factor)
            crop_y = int(image_source.shape[0] / crop_factor)    
            adjust_x = int(crop_x * (crop_factor - 1))
            adjust_y = int(crop_y * crop_factor)
            image_source = image_source[0:adjust_y, crop_x:adjust_x]
        image = cv2.addWeighted(image_source, contrast, image_source, 0, brightness)
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        image = simplest_cb(image, 50)
        resize_factor = (resize_height / image.shape[0])
        if (resize_height > 0):
            width = int(image.shape[1] * resize_factor)
            height = int(image.shape[0] * resize_factor)
            dimensions = (width, height)
            image = cv2.resize(image, dimensions, interpolation = cv2.INTER_CUBIC)
        if (blur_factor > 0):
            image = cv2.blur(image, (blur_factor, blur_factor))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)
        results = face_detection(image, crop_x, crop_y, filename, extension, resize_factor, biggest)
        if results is not False:
            return results

    return default

################################################################################
# The 'apply_threshold' function used by the 'simplest_cb' function.
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

################################################################################
# The 'apply_threshold' function used by the 'simplest_cb' function.
def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

################################################################################
# The 'simplest_cb' function.
# SOURCE: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil( n_cols * (1.0 - half_percent))]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

################################################################################
# The 'face_detection' function.
def face_detection(image_source, crop_x, crop_y, filename, extension, resize_factor = 1, biggest = False):

    ############################################################################
    # Initialize the counter stuff.
    counter = 0
    # rotation_max = 4
    rotation_max = 1
    border_size = 0

    ############################################################################
    # Set the CV2 flags.
    flags = cv2.CASCADE_DO_CANNY_PRUNING

    ############################################################################
    # If we are looking for the biggest face, set that flag.
    if biggest:
        flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT

    ############################################################################
    # Set a copy of the image source.
    image = image_source

    ############################################################################
    # Roll through the rotations to use.
    while counter < rotation_max:

        ########################################################################
        # Set borders for better face detection.
        border_color = int(image_source[int(image_source.shape[0]/4),0])
        border_x_size = int(0.50 * image_source.shape[1])
        border_y_size = int(0.10 * image_source.shape[0])
        image = cv2.copyMakeBorder(image_source, 0, border_y_size, border_x_size, border_x_size, cv2.BORDER_CONSTANT, None, border_color)

        ########################################################################
        # Set the min and max image size.
        side = math.sqrt(image.size)
        min_length = int(side / 20)
        max_length = int(side / 2)

        ########################################################################
        # Try and find faces.
        faces_found = CASCADES_LOADED['ALT2'].detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = CASCADES_LOADED['DEFAULT'].detectMultiScale(image, 1.4, 6, flags, (min_length, min_length), (max_length, max_length))
        if len(faces_found) == 0:
            faces_found = CASCADES_LOADED['DEFAULT'].detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))

        ########################################################################
        # TODO: Some simple debugging. Don't use Python to do image writing.
        # Instead use the output with a batch processor like ImageMagick.
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
        # If a face is found, return the values. Or else, multiply the counter
        # by 90 to get the number of degrees the image should be rotated.
        if (len(faces_found) > 0):
            rotation = counter * 90
            final_x = int(int((faces_found[0][0] - border_x_size) + crop_x) / resize_factor)
            final_y = int((faces_found[0][1] - border_y_size) / resize_factor)
            final_w = int(faces_found[0][2] / resize_factor)
            final_h = int(faces_found[0][3] / resize_factor)
            final = {
                'x': final_x,
                'y': final_y,
                'w': final_w,
                'h': final_h,
                'd': int(rotation),
            }
            return final
        else:
            image_source = cv2.rotate(image_source, cv2.ROTATE_90_CLOCKWISE)
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
    for k, v in CASCADES_DICTIONARY.items():
        v = os.path.join(data_dir, v)
        try:
            if not os.path.exists(v):
                raise cv2.error('no such file')
            CASCADES_LOADED[k] = cv2.CascadeClassifier(v)
        except cv2.error:
            fatal("cannot load {} from {}".format(k, v))

################################################################################
# The '__main__' function.
def __main__():
    ap = argparse.ArgumentParser(description='A face detector script for batch processing')
    ap.add_argument('--biggest', action="store_true",
                    help='Extract only the biggest face')
    ap.add_argument('file', help='Input image file')
    args = ap.parse_args()

    ############################################################################
    # And here's where we invoke it and get the the output.
    load_cascades(DATA_DIRECTORY)

    ############################################################################
    # And here's where we invoke it and get the the output.
    image_data = manage_face_detection(args.file, args.biggest)

    ############################################################################
    # Get the rotation from the image data.
    rotation = int(image_data['d'])

    ############################################################################
    # Set the image data string.
    image_data_string = ' ' . join(str(value) for value in image_data.values())

    ############################################################################
    # Return the final return value.
    print (image_data_string)

if __name__ == '__main__':
    sys.exit(__main__())

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
