# Facedetect CV2

## Prerequisites

If you get an error like this:

	ModuleNotFoundError: No module named 'cv2'

That simply means that the Python OpenCV stuff is not installed. Install specific versions of `numpy` and `opencv-python` like this:

	pip3 install --break-system-packages -r requirements.txt

***

Or install `opencv-python` directly like this:

	pip3 install opencv-python --verbose

Might need to specify a version:

	pip3 install --break-system-packages opencv-python==4.1.2.30 --verbose
	pip3 install --break-system-packages opencv-python==4.3.0.38 --verbose

	pip3 install --break-system-packages opencv-python==4.5.5.64 --verbose
	pip3 install --break-system-packages opencv-python==4.11.0.86 --verbose
