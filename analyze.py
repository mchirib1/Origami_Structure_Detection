import argparse
import sys
import os

import time
from datetime import date

from utils.analysis_utils import DataAnalyzer


def analyze():
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

    # counts the number of annotations in each image of the stack
    # also sums the total number of annotations in the stack
    if opts.count_anns:
        ana = DataAnalyzer(src)
        ana.count_annotations(dst)

    # similar to count_annotations method except here the wording is changed to denote these are predictions
    if opts.count_preds:
        ana = DataAnalyzer(src)
        ana.count_predictions(dst)

    # similar to the above methods except here the pixel array shape is being counted
    if opts.count_dims:
        ana = DataAnalyzer(src)
        ana.count_image_dimensions(dst)

    # plots bounding box distributions
    if opts.scat_hist:
        ana = DataAnalyzer(src)
        ana.data_scatter_hist(dst)


parser = argparse.ArgumentParser(description='CLI tool for augmenting source images and corresponding annotations beore feeding into the YOLO machine learning framework.')
parser.add_argument('--src', type=str, default=None, metavar='source_path', help='Path to source directory containing image and label stack')
parser.add_argument('--dst', type=str, default=None, metavar='dest_path', help='Path to create a destination directory storing output')
parser.add_argument('--count_anns', default=False, action='store_true', help='Counts the number of annotations in each image as well as total for stack')
parser.add_argument('--count_preds', default=False, action='store_true', help='Counts the number of predictions in each image as well as total for stack')
parser.add_argument('--count_dims', default=False, action='store_true', help='Counts pixel array shape for each image in the stack')
parser.add_argument('--scat_hist', default=False, action='store_true', help='Plots bounding box distributions')
opts = parser.parse_args()

if __name__ == '__main__':
    begin = time.time()
    today = date.today().strftime('%Y%m%d')

    analyze()

    fin = time.time() - begin
    print(f'Code completed in {fin:0.4f} seconds on {today}.')
