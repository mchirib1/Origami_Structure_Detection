import argparse
import sys
import os

import time
from datetime import date

from Utils.Augmentation_utils import DataAugmenter


def augment():
    print(opts)

    src = f'{os.getcwd()}/{opts.src}'
    dst = f'{os.getcwd()}/{opts.dst}'

    # confirm src directory
    if opts.src is None:
        print('Directory not found or none specified please try again.')
        sys.exit(0)

    # confirm dst directory
    if opts.dst is None:
        print('Directory not found or none specified please try again.')
        sys.exit(0)

    # Convert from full image to data only in upper left hand quadrant
    if opts.cvt_gray:
        aug = DataAugmenter(src)
        aug.cvt_grayscale(dst)

    # edit the contrast and brightness by scaling alpha and beta
    if opts.cvt_ab:
        aug = DataAugmenter(src)

        alpha = float(opts.cvt_ab[0])
        beta = int(opts.cvt_ab[1])

        aug.cvt_alpha_beta(alpha, beta, dst)

    # randomly rotates an image by angle between 1 and theta, n times
    if opts.rot:
        aug = DataAugmenter(src)

        n = int(opts.rot[0])
        theta = int(opts.rot[1])

        aug.rotate_img(n, theta, dst)

    # these two methods zero out certain quadrants of the image changing the density of the data within the image
    if opts.quad1:
        aug = DataAugmenter(src)
        aug.quadrant_1(dst) # sets all pixels outside upper left quadrant to 0

    if opts.quad3:
        aug = DataAugmenter(src)
        aug.quadrant_3(dst) # sets all pixels outside lower right quadrant to 0


parser = argparse.ArgumentParser(description='CLI tool for augmenting source images and corresponding annotations beore feeding into the YOLO machine learning framework.')
parser.add_argument('--src', type=str, default=None, metavar='source_path', help='Path to source directory containing image and label stack')
parser.add_argument('--dst', type=str, default=None, metavar='dest_path', help='Path to create a destination directory storing output')
parser.add_argument('--cvt_gray', default=False, action='store_true', help='Converts image stack to grayscale')
parser.add_argument('--cvt_ab', default=False, nargs=2, metavar=('alpha', 'beta'), help='Edits constants and brightness by scaling alpha and beta')
parser.add_argument('--rot', default=False, nargs=2, metavar=('n', 'theta'), help='Rotates each image and corresponding label by random angle between 1 and theta, n times')
parser.add_argument('--quad1', default=False, action='store_true', help='Sets all pixels outside of the upper left quadrant to zero')
parser.add_argument('--quad3', default=False, action='store_true', help='Sets all pixels outside of the lower right quadrant to zero')
opts = parser.parse_args()


if __name__ == '__main__':
    begin = time.time()
    today = date.today().strftime('%Y%m%d')

    augment()

    fin = time.time() - begin
    print(f'Code completed in {fin:0.4f} seconds on {today}.')
