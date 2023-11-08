# Facedetect CV2

## Prerequisites

If you get an error like this:

	ModuleNotFoundError: No module named 'cv2'

That simply means that the Python OpenCV stuff is not installed. Install it like this:

	pip3 install opencv-python --verbose

Might need to specify a version:

	pip3 install opencv-python==4.1.2.30 --verbose

### More Notes

I don’t know why. But these work on Apple Silicon.

	pip3 install opencv-python==4.5.5.64 --verbose
	pip3 install opencv-python==4.8.1.78 --verbose
	python3 -m pip install opencv-python	
