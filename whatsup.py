#!/usr/bin/env python3

################################################################################
# __        ___           _
# \ \      / / |__   __ _| |_ ___ _   _ _ __
#  \ \ /\ / /| '_ \ / _` | __/ __| | | | '_ \
#   \ V  V / | | | | (_| | |_\__ \ |_| | |_) |
#    \_/\_/  |_| |_|\__,_|\__|___/\__,_| .__/
#                                      |_|
#
# 2020-04-30: An updated version of Stuporglue’s “Whatsup” script. Now uses
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
# 	https://stuporglue.org/automatically-orient-scanned-photos-correctly-with-opencv/
#
# Two other — now dead — sites are referenced as a source as well:
# 	http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
# 	http://opencv.willowgarage.com/documentation/python/core_operations_on_arrays.html#createmat
#
################################################################################

################################################################################
# Import various modules and functions.
from __future__ import print_function, division, generators, unicode_literals
import sys
import os
import cv2
import math
import numpy as np;
import pathlib

################################################################################
# CV compatibility stubs
if 'IMREAD_GRAYSCALE' not in dir(cv2):
	# <2.4
	cv2.IMREAD_GRAYSCALE = 0
if 'cv' in dir(cv2):
	# <3.0
	cv2.CASCADE_DO_CANNY_PRUNING = cv2.cv.CV_HAAR_DO_CANNY_PRUNING
	cv2.CASCADE_FIND_BIGGEST_OBJECT = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
	cv2.FONT_HERSHEY_SIMPLEX = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, cv2.cv.CV_AA)
	cv2.LINE_AA = cv2.cv.CV_AA

############################################################################
# Set the cascade data directory and related stuff.
DATA_DIRECTORY = '/usr/local/lib/python3.7/site-packages/cv2/data/'
# CASCADES_TO_USE = ('haarcascade_frontalface_alt.xml', 'haarcascade_profileface.xml', 'haarcascade_fullbody.xml')
CASCADES_TO_USE = ('haarcascade_profileface.xml', 'haarcascade_fullbody.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_default.xml')

################################################################################
# The 'detectFaces' function.
def detectFaces(image, cc, filename, extension, biggest=False):

	############################################################################
	# Initialize the counter.
	counter = 0

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
	while counter < 4:

		########################################################################
		# Attempt to detect some faces.
		faces_detected = cc.detectMultiScale(image, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))

		########################################################################
		# If a face is found, multiply the counter by 90 to get the number of degrees the image should be rotated.
		if (len(faces_detected) > 0):
			rotation = counter * 90
			return rotation

		########################################################################
		# Rotate the image 90 degrees clockwise.
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

		########################################################################
		# Increment the counter.
		counter = counter + 1

	return False

################################################################################
# The 'detectBrightest' function.
def detectBrightest(image, filename, extension):

	############################################################################
	# Get the dimensions of the image.
	image_h, image_w = image.shape[:2]

	############################################################################
	# Get the slices for the top, right, bottom and left regions for analysis.
	sample_top = image[0:round(image_h/3), 0:image_w]
	sample_left = image[0:image_h, 0:round(image_w/3)]
	sample_bottom = image[round(2*(image_h/3)):image_h, 0:image_w]
	sample_right = image[0:image_h, round(2*(image_w/3)):image_w]

	####################################################################
	# Resize the sample images to 10x10 to average things out
	resize = (10, 10)
	sample_top = cv2.resize(sample_top, resize, interpolation = cv2.INTER_CUBIC)
	sample_left = cv2.resize(sample_left, resize, interpolation = cv2.INTER_CUBIC)
	sample_bottom = cv2.resize(sample_bottom, resize, interpolation = cv2.INTER_CUBIC)
	sample_right = cv2.resize(sample_right, resize, interpolation = cv2.INTER_CUBIC)

	####################################################################
	# Blur the images to even further average things out.
	sample_top = cv2.GaussianBlur(sample_top, (5,5), cv2.BORDER_DEFAULT)
	sample_left = cv2.GaussianBlur(sample_left, (5,5), cv2.BORDER_DEFAULT)
	sample_bottom = cv2.GaussianBlur(sample_bottom, (5,5), cv2.BORDER_DEFAULT)
	sample_right = cv2.GaussianBlur(sample_right, (5,5), cv2.BORDER_DEFAULT)

	####################################################################
	# Build a mapping of those samples.
	sides = {}
	sides['top'] = cv2.mean(sample_top)[0]
	sides['left'] = cv2.mean(sample_left)[0]
	sides['bottom'] = cv2.mean(sample_bottom)[0]
	sides['right'] = cv2.mean(sample_right)[0]

	####################################################################
	# Get the max value from the sides.
	max_side = max(sides, key = sides.get)

	####################################################################
	# Set the mapping for rotation values.
	rotation = { 'top': 0, 'left': 90, 'bottom': 180, 'right': 270 }

	return rotation[max_side]

################################################################################
# The 'tryDetect' function.
def tryDetect(biggest=False):

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
	# Roll through the cascades.
	for THIS_CASCADE in CASCADES_TO_USE:

		########################################################################
		# Initialize the counter.
		counter = 4

		########################################################################
		# Define the cascade classifier.
		cc = cv2.CascadeClassifier(os.path.join(DATA_DIRECTORY, THIS_CASCADE))

		########################################################################
		# Roll through the sizes.
		while counter > 0:

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
			# Send the image to the 'dectectFaces' method.
			results = detectFaces(image_resized, cc, filename, extension, biggest)

			####################################################################
			# If we have results return the results.
			if False and results is not False:
				return results

			counter = counter - 1

	############################################################################
	# If no faces are found, use the brightest side for orientation instead.
	return detectBrightest(image, filename, extension)
	# return 0

################################################################################
# Usage Check
if ((len(sys.argv) != 2 and len(sys.argv) != 3) or (len(sys.argv) == 3)):
	print ("USAGE: whatsup filename")
	sys.exit(-1)

################################################################################
# Sanity check
if not os.path.isfile(sys.argv[-1]):
	print ("File '" + sys.argv[-1] + "' not found.")
	sys.exit(-1)

################################################################################
# Make it happen
print (str(tryDetect(True)))
