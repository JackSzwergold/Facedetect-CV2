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

	counter = 0

	side = math.sqrt(image_resized.size)
	minlen = int(side / 20)
	maxlen = int(side / 2)
	flags = cv2.CASCADE_DO_CANNY_PRUNING
	# flags = cv2.CASCADE_SCALE_IMAGE

	# 4 shots at getting faces.
	while counter < 4:
		faces = cc.detectMultiScale(image_resized, 1.3, 6, flags, (minlen, minlen), (maxlen, maxlen))
		print(faces)
		if (len(faces) > 0):

			return counter * 90

		# The rotation routine:
		# tmp_mat = cv2.GetMat(image_resized)
		# kernel = np.ones((5,5),np.float32)/25
		# small_img_filtered = cv2.filter2D(image_resized, cv2.CV_8U, kernel.transpose())
		small_img_blur = cv2.GaussianBlur(image_resized, (5, 5), 0)
		small_img_filtered, dimensions = cv2.threshold(small_img_blur, 0, 255, cv2.THRESH_BINARY)
		tmp_mat = cv2.moments(small_img_filtered, 0)

		print(tmp_mat)
		# exit()
		#
		# # Create a Mat that is rotated 90 degrees in size (3x4 becomes 4x3)
		# # tmp_dst_mat = cv2.CreateMat(tmp_mat.cols, tmp_mat.rows, cv2.CV_8UC1)
		# tmp_dst_mat = np.zeros((tmp_mat.rows, tmp_mat.cols, 3), np.uint8)
		#
		# # Create a Mat that is rotated 90 degrees in size (3x4 becomes 4x3)
		# # dst_mat = cv2.CreateMat(tmp_mat.cols, tmp_mat.rows, cv2.CV_8UC1)
		# dst_mat = np.zeros((tmp_mat.rows, tmp_mat.cols, 3), np.uint8)
		#
		# # To rotate 90 clockwise, we transpose, then flip on Y axis
		# cv2.transpose(image_resized, tmp_dst_mat) # Transpose it
		# cv2.flip(tmp_dst_mat, dst_mat, flipMode= 1) # flip it
		#
		# # put it back in image_resized so we can try to detect faces again
		# image_resized = cv2.getImage(dst_mat)

		# Increment the counter.
		counter = counter + 1

	return False


# Detect which side of the photo is brightest. Hopefully it will be the sky.
def detectBrightest(image):
	image_scale = 4 # This scale factor doesn't matter much. It just gives us less pixels to iterate over later
	newsize = (cv2.Round(image.width/image_scale), cv2.Round(image.height/image_scale)) # find new size
	image_resized = cv2.CreateImage(newsize, 8, 1)
	cv2.Resize(image, image_resized, cv2.CV_INTER_LINEAR)

	# Take the top 1/3, right 1/3, etc. to compare for brightness
	width = image_resized.width
	height = image_resized.height
	top = image_resized[0:height/3, 0:width]
	right = image_resized[0:height, (width/3*2):width]
	left = image_resized[0:height, 0:width/3]
	bottom = image_resized[(height/3*2):height, 0:height]

	sides = {'top':top, 'left':left, 'bottom':bottom, 'right':right}

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

# Defining the 'tryDetect' method.
def tryDetect():

	# Set the filename from the input argument.
	filename = sys.argv[-1]

	# Set the image path.
	image_path = os.path.abspath(filename)

	# print(os.path.dirname(image_path))
	print(os.path.basename(image_path))
	# print(filename)
	# print(pathlib.Path(image_path).suffix)
	print(pathlib.Path(image_path).suffix)
	exit()

	# Load the image into the scriupt.
	cv2.IMREAD_GRAYSCALE = 0
	source_img = cv2.imread(image_path) # the image itself

	data_directory = '/usr/local/lib/python3.7/site-packages/cv2/data/'
	cascades_to_use = ('haarcascade_frontalface_alt.xml', 'haarcascade_profileface.xml', 'haarcascade_fullbody.xml')

	for this_cascade in cascades_to_use:

		# Define the cascade classifier.
		cc = cv2.CascadeClassifier(os.path.join(data_directory, this_cascade))

		image_scale = 4
		while image_scale > 0:

			# Get the dimensions of the image.
			img_shape = np.shape(source_img)
			img_w = img_shape[0]
			img_h = img_shape[1]

			# Calculate the new size for the images.
			new_w = round(img_w / image_scale)
			new_h = round(img_h / image_scale)
			newsize = (new_h, new_w)

			# Resize the image.
			image_resized = cv2.resize(source_img, newsize, interpolation = cv2.INTER_CUBIC)

			# Write the image for debugging.
			cv2.imwrite('test_' + str(new_w) + 'x' + str(new_h) + '.jpg', image_resized)

			# Send the image to the 'dectectFaces' method.
			results = detectFaces(image_resized, cc)

			if results is not False:
				return results

			image_scale = image_scale - 1

	# no faces found, use the brightest side for orientation instead
	# return detectBrightest(source_img)
	return


# Usage Check
if ((len(sys.argv) != 2 and len(sys.argv) != 3) or (len(sys.argv) == 3 and sys.argv[1] != '--debug')):
	print ("USAGE: whatsup filename")
	sys.exit(-1)

# Sanity check
if not os.path.isfile(sys.argv[-1]):
	print ("File '" + sys.argv[-1] + "' does not exist")
	sys.exit(-1)

# Make it happen
print (str(tryDetect()))
