#!/usr/bin/env python3

################################################################################
# __        ___           _
# \ \      / / |__   __ _| |_ ___ _   _ _ __
#  \ \ /\ / /| '_ \ / _` | __/ __| | | | '_ \
#   \ V  V / | | | | (_| | |_\__ \ |_| | |_) |
#    \_/\_/  |_| |_|\__,_|\__|___/\__,_| .__/
#                                      |_|
#
# 2020-04-30: An updated version of Stuporglue’s “Whatsup” script.
# Source: https://stuporglue.org/automatically-orient-scanned-photos-correctly-with-opencv/
################################################################################

################################################################################
# This script reads in a file and counter to determine which orientation is correct
# by looking for faces in the photos
# It starts with the existing orientation, then rotates it 90 degrees at a time until
# it has either tried all 4 directions or until it finds a face

################################################################################
# INSTALL: Put the xml files in /usr/local/share, or change the script. Put whatsup somewhere in your path

################################################################################
# Usage: whatsup filename
# Returns the number of degrees it should be rotated clockwise to orient the faces correctly

################################################################################
# Some code came from here: http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
# The rest was cobbled together by me from the documentation here [1] and from snippets and samples found via Google
# [1] http://opencv.willowgarage.com/documentation/python/core_operations_on_arrays.html#createmat

import sys
import os
import cv2
import math
import numpy as np;
import pathlib

def detectFaces(image_resized, cc):

	############################################################################
	# Initialize the counter.
	counter = 0

	############################################################################
	# Set the min and max image size.
	side = math.sqrt(image_resized.size)
	min_length = int(side / 20)
	max_length = int(side / 2)

	# rotations_to_use = ('ROTATE_90_CLOCKWISE', 'ROTATE_90_COUNTERCLOCKWISE', 'ROTATE_180')
	rotations_to_use = ('ROTATE_90_CLOCKWISE', 'ROTATE_90_COUNTERCLOCKWISE')

	############################################################################
	# Set the CV2 flags.
	flags = cv2.CASCADE_DO_CANNY_PRUNING
	# flags = cv2.CASCADE_SCALE_IMAGE

	############################################################################
	# Roll through the rotations to use.
	for this_rotation in rotations_to_use:

		# Attempt to detect some faces.
		faces = cc.detectMultiScale(image_resized, 1.3, 6, flags, (min_length, min_length), (max_length, max_length))

		########################################################################
		# TODO: Debugging.
		print(this_rotation)

		########################################################################
		# If a face is found, do this.
		if (len(faces) > 0):
			return counter * 90

		########################################################################
		# Rotate the image.
		small_image_blur = cv2.GaussianBlur(image_resized, (5, 5), 0)
		small_image_filtered, dimensions = cv2.threshold(small_image_blur, 0, 255, cv2.THRESH_BINARY)

		########################################################################
		# Increment the counter.
		counter = counter + 1

	return False

################################################################################
# Detect which side of the photo is brightest. Hopefully it will be the sky.
def detectBrightest(image):
	image_scale = 4 # This scale factor doesn't matter much. It just gives us less pixels to iterate over later
	newsize = (cv2.Round(image.width/image_scale), cv2.Round(image.height/image_scale)) # find new size
	image_resized = cv2.CreateImage(newsize, 8, 1)
	cv2.Resize(image, image_resized, cv2.CV_INTER_LINEAR)

	############################################################################
	# Take the top 1/3, right 1/3, etc. to compare for brightness
	width = image_resized.width
	height = image_resized.height
	top = image_resized[0:height/3, 0:width]
	right = image_resized[0:height, (width/3*2):width]
	left = image_resized[0:height, 0:width/3]
	bottom = image_resized[(height/3*2):height, 0:height]

	sides = {'top':top, 'left':left, 'bottom':bottom, 'right':right}

	############################################################################
	# Find the brightest side
	greatest = 0
	winning = 'top'
	for name in sides:
		sidelum = 0
		side = sides[name]
		for x in range(side.rows - 1):
			for y in range(side.cols - 1):
				sidelum = sidelum + side[x, y]
		sidelum = sidelum/(side.rows*side.cols)
		if sidelum > greatest:
			winning = name

	cv2.Rectangle(image_resized, first, second, cv2.RGB(125, 125, 125), 3, 8, 0)
	cv2.NamedWindow("Faces")
	cv2.ShowImage("Faces", image_resized)
	cv2.WaitKey(3000)

	returns = {'top':0, 'left':90, 'bottom':180, 'right':270}

	return returns[winning]

################################################################################
# Defining the 'tryDetect' method.
def tryDetect():

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
	# Load the image into the scriupt.
	cv2.IMREAD_GRAYSCALE = 0
	source_img = cv2.imread(image_path) # the image itself

	data_directory = '/usr/local/lib/python3.7/site-packages/cv2/data/'
	cascades_to_use = ('haarcascade_frontalface_alt.xml', 'haarcascade_profileface.xml', 'haarcascade_fullbody.xml')

	for this_cascade in cascades_to_use:

		########################################################################
		# Define the cascade classifier.
		cc = cv2.CascadeClassifier(os.path.join(data_directory, this_cascade))

		image_scale = 4
		while image_scale > 0:

			####################################################################
			# Get the dimensions of the image.
			img_shape = np.shape(source_img)
			image_w = img_shape[0]
			image_h = img_shape[1]

			####################################################################
			# Calculate the new size for the images.
			resize_w = round(image_w / image_scale)
			resize_h = round(image_h / image_scale)

			####################################################################
			# Resize the image.
			image_resized = cv2.resize(source_img, (resize_h, resize_w), interpolation = cv2.INTER_CUBIC)

			####################################################################
			# Write the image for debugging.
			image_test = filename + '_' + str(resize_w) + 'x' + str(resize_h) + extension
			cv2.imwrite(image_test, image_resized)

			####################################################################
			# Send the image to the 'dectectFaces' method.
			results = detectFaces(image_resized, cc)

			if results is not False:
				return results

			image_scale = image_scale - 1

	############################################################################
	# no faces found, use the brightest side for orientation instead
	# return detectBrightest(source_img)
	return

################################################################################
# Usage Check
if ((len(sys.argv) != 2 and len(sys.argv) != 3) or (len(sys.argv) == 3)):
	print ("USAGE: whatsup filename")
	sys.exit(-1)
################################################################################

# Sanity check
if not os.path.isfile(sys.argv[-1]):
	print ("File '" + sys.argv[-1] + "' does not exist")
	sys.exit(-1)

################################################################################
# Make it happen
print (str(tryDetect()))
