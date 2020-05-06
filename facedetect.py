#!/usr/bin/env python3

################################################################################
# __        ___           _
# \ \      / / |__   __ _| |_ ___ _   _ _ __
#  \ \ /\ / /| '_ \ / _` | __/ __| | | | '_ \
#   \ V  V / | | | | (_| | |_\__ \ |_| | |_) |
#    \_/\_/  |_| |_|\__,_|\__|___/\__,_| .__/
#                                      |_|
#
# 2020-05-04: An updated version of uri D'Elia’s “Facedetect” script. Now uses
# Python3 and CV2 methods and conventions.
#
# Usage: facedetect [filepath]
#
# Output: x y width height
#
################################################################################

################################################################################
# facedetect: a simple face detector for batch processing
# Copyright(c) 2013-2017 by wave++ "Yuri D'Elia" <wavexx@thregr.org>
# Distributed under GPLv2+ (see COPYING) WITHOUT ANY WARRANTY.
#
# Primary source:
# 	https://www.thregr.org/~wavexx/software/facedetect/
#
################################################################################

################################################################################
# Import various modules and functions.
from __future__ import print_function, division, generators, unicode_literals

import argparse
import numpy as np
import cv2
import math
import sys
import os

################################################################################
# Set the data directory root.
# DATA_DIRECTORY = '/usr/share/opencv/'
DATA_DIRECTORY = cv2.data.haarcascades

################################################################################
# Define the profiles.
PROFILES = {
    'haarcascade_frontalface_alt2.xml': { 'scaleFactor': 1.3, 'minNeighbors': 6 },
    'haarcascade_frontalface_default.xml': { 'scaleFactor': 1.4, 'minNeighbors': 6 },
}

################################################################################
# Init the cascahes.
CASCADES = {}

################################################################################
# Face normalization values.
NORM_SIZE = 100
NORM_MARGIN = 10

################################################################################
# The 'error' function.
def error(msg):
    sys.stderr.write("{}: error: {}\n".format(os.path.basename(sys.argv[0]), msg))

################################################################################
# The 'fatal' function.
def fatal(msg):
    error(msg)
    sys.exit(1)

################################################################################
# The 'load_cascades' function.
def load_cascades(data_dir):
    for key, value in PROFILES.items():
        key_full = os.path.join(data_dir, key)
        try:
            if not os.path.exists(key_full):
                raise cv2.error('no such file')
            CASCADES[key] = cv2.CascadeClassifier(key_full)
        except cv2.error:
            fatal("cannot load {}".format(key))

################################################################################
# The 'crop_rect' function.
def crop_rect(image, rect, shave=0):
    return image[rect[1]+shave:rect[1]+rect[3]-shave,
              rect[0]+shave:rect[0]+rect[2]-shave]

################################################################################
# The 'crop_rect' function.
def shave_margin(image, margin):
    return image[margin:-margin, margin:-margin]

################################################################################
# The 'norm_rect' function.
def norm_rect(image, rect, equalize=True, same_aspect=False):
    roi = crop_rect(image, rect)
    if equalize:
        roi = cv2.equalizeHist(roi)
    side = NORM_SIZE + NORM_MARGIN
    if same_aspect:
        scale = side / max(rect[2], rect[3])
        dsize = (int(rect[2] * scale), int(rect[3] * scale))
    else:
        dsize = (side, side)
    roi = cv2.resize(roi, dsize, interpolation=cv2.INTER_CUBIC)
    return shave_margin(roi, NORM_MARGIN)

################################################################################
# The 'rank' function.
def rank(image, rects):
    scores = []
    best = None

    for i in range(len(rects)):
        rect = rects[i]
        roi_n = norm_rect(image, rect, equalize=False, same_aspect=True)
        roi_l = cv2.Laplacian(roi_n, cv2.CV_8U)
        e = np.sum(roi_l) / (roi_n.shape[0] * roi_n.shape[1])

        dx = image.shape[1] / 2 - rect[0] + rect[2] / 2
        dy = image.shape[0] / 2 - rect[1] + rect[3] / 2
        d = math.sqrt(dx ** 2 + dy ** 2) / (max(image.shape) / 2)

        s = (rect[2] + rect[3]) / 2
        scores.append({'s': s, 'e': e, 'd': d})

    sMax = max([x['s'] for x in scores])
    eMax = max([x['e'] for x in scores])

    for i in range(len(scores)):
        s = scores[i]
        sN = s['sN'] = s['s'] / sMax
        eN = s['eN'] = s['e'] / eMax
        f = s['f'] = eN * 0.7 + (1 - s['d']) * 0.1 + sN * 0.2

    ranks = range(len(scores))
    ranks = sorted(ranks, reverse=True, key=lambda x: scores[x]['f'])
    for i in range(len(scores)):
        scores[ranks[i]]['RANK'] = i

    return scores, ranks[0]

################################################################################
# The 'mssim_norm' function.
def mssim_norm(X, Y, K1=0.01, K2=0.03, win_size=11, sigma=1.5):
    C1 = K1 ** 2
    C2 = K2 ** 2
    cov_norm = win_size ** 2

    ux = cv2.GaussianBlur(X, (win_size, win_size), sigma)
    uy = cv2.GaussianBlur(Y, (win_size, win_size), sigma)
    uxx = cv2.GaussianBlur(X * X, (win_size, win_size), sigma)
    uyy = cv2.GaussianBlur(Y * Y, (win_size, win_size), sigma)
    uxy = cv2.GaussianBlur(X * Y, (win_size, win_size), sigma)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    A1 = 2 * ux * uy + C1
    A2 = 2 * vxy + C2
    B1 = ux ** 2 + uy ** 2 + C1
    B2 = vx + vy + C2
    D = B1 * B2
    S = (A1 * A2) / D

    return np.mean(shave_margin(S, (win_size - 1) // 2))

################################################################################
# The 'face_detect' function.
def face_detect(image, biggest=False):

    ############################################################################
    # Set some values.
    side = math.sqrt(image.size)
    minlen = int(side / 20)
    maxlen = int(side / 2)

    ############################################################################
    # Set some flags.
    flags = cv2.CASCADE_DO_CANNY_PRUNING
    if biggest:
        flags |= cv2.CASCADE_FIND_BIGGEST_OBJECT

    ############################################################################
    # Convert the image to grayscale.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)

    ############################################################################
    # Roll through the cascades and try to detect some faces.
    for cc_key in CASCADES:
        cc = CASCADES[cc_key]
        scaleFactor = PROFILES[cc_key]['scaleFactor']
        minNeighbors = PROFILES[cc_key]['minNeighbors']
        results = cc.detectMultiScale(image, scaleFactor, minNeighbors, flags, (minlen, minlen), (maxlen, maxlen))
        if len(results) > 0:
            return results
    return results

################################################################################
# The 'face_detect_file' function.
def face_detect_file(path, biggest=False):
    image = cv2.imread(path)
    if image is None:
        fatal("cannot load input image {}".format(path))
    features = face_detect(image, biggest)
    return image, features

################################################################################
# The 'pairwise_similarity' function.
def pairwise_similarity(image, features, template, **mssim_args):
    template = np.float32(template) / 255
    for rect in features:
        roi = norm_rect(image, rect)
        roi = np.float32(roi) / 255
        yield mssim_norm(roi, template, **mssim_args)

################################################################################
# The '__main__' function.
def __main__():
    argument_parser = argparse.ArgumentParser(description='A simple face detector for batch processing')
    argument_parser.add_argument('--biggest', action="store_true",
                    help='Extract only the biggest face')
    argument_parser.add_argument('--best', action="store_true",
                    help='Extract only the best matching face')
    argument_parser.add_argument('-c', '--center', action="store_true",
                    help='Print only the center coordinates')
    argument_parser.add_argument('--data-dir', metavar='DIRECTORY', default=DATA_DIRECTORY,
                    help='OpenCV data files directory')
    argument_parser.add_argument('-q', '--query', action="store_true",
                    help='Query only (exit 0: face detected, 2: no detection)')
    argument_parser.add_argument('-s', '--search', metavar='FILE',
                    help='Search for faces similar to the one supplied in FILE')
    argument_parser.add_argument('--search-threshold', metavar='PERCENT', type=int, default=30,
                    help='Face similarity threshold (default: 30%%)')
    argument_parser.add_argument('-o', '--output', help='Image output file')
    argument_parser.add_argument('-d', '--debug', action="store_true",
                    help='Add debugging metrics in the image output file')
    argument_parser.add_argument('file', help='Input image file')
    args = argument_parser.parse_args()

    ############################################################################
    # Load the cascades.
    load_cascades(args.data_dir)

    ############################################################################
    # Detect faces in input image.
    image, features = face_detect_file(args.file, args.query or args.biggest)

    ############################################################################
    # Match against the requested face.
    sim_scores = None
    if args.search:
        s_im, s_features = face_detect_file(args.search, True)
        if len(s_features) == 0:
            fatal("cannot detect face in template")
        sim_scores = []
        sim_features = []
        sim_threshold = args.search_threshold / 100
        sim_template = norm_rect(s_im, s_features[0])
        for i, score in enumerate(pairwise_similarity(image, features, sim_template)):
            if score >= sim_threshold:
                sim_scores.append(score)
                sim_features.append(features[i])
        features = sim_features

    ############################################################################
    # Exit early if possible.
    if args.query:
        return 0 if len(features) else 2

    ############################################################################
    # Compute scores.
    scores = []
    best = None
    if len(features) and (args.debug or args.best or args.biggest or sim_scores):
        scores, best = rank(image, features)
        if sim_scores:
            for i in range(len(features)):
                scores[i]['MSSIM'] = sim_scores[i]

    ############################################################################
    # Debug features.
    if args.output:
        image = cv2.imread(args.file)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontHeight = cv2.getTextSize("", font, 0.5, 1)[0][1] + 5

        for i in range(len(features)):
            if best is not None and i != best and not args.debug:
                next

            rect = features[i]
            fg = (0, 255, 255) if i == best else (255, 255, 255)

            xy1 = (rect[0], rect[1])
            xy2 = (rect[0] + rect[2], rect[1] + rect[3])
            cv2.rectangle(image, xy1, xy2, (0, 0, 0), 4)
            cv2.rectangle(image, xy1, xy2, fg, 2)

            if args.debug:
                lines = []
                for k, v in scores[i].items():
                    lines.append("{}: {}".format(k, v))
                h = rect[1] + rect[3] + fontHeight
                for line in lines:
                    cv2.putText(image, line, (rect[0], h), font, 0.5, fg, 1, cv2.LINE_AA)
                    h += fontHeight

        cv2.imwrite(args.output, image)

    ############################################################################
    # Output.
    if (args.best or args.biggest) and best is not None:
        features = [features[best]]

    if args.center:
        for rect in features:
            x = int(rect[0] + rect[2] / 2)
            y = int(rect[1] + rect[3] / 2)
            print("{} {}".format(x, y))
    else:
        for rect in features:
            print("{} {} {} {}".format(*rect))

    return 0

################################################################################
# And here’s where we invoke it and get the the output.
if __name__ == '__main__':
    sys.exit(__main__())
