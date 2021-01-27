import os
import time
from datetime import date

import cv2
from matplotlib import pyplot as plt

class DataAnalyzer:
    '''A class to help with data augmentation of the triangle image analysis data.  Handles things like numerical
    augmentations as well as rotations.'''

    def __init__(self, src_dir):
        self.src_dir = src_dir

        self.images = [file for file in os.listdir(src_dir) if
                       file.split(".")[1] == 'png' or file.split(".")[1] == 'jpg']
        self.labels = [file for file in os.listdir(src_dir) if file.split(".")[1] == 'txt'
                       and file != 'classes.txt']

        self.date = date.today().strftime('%Y%m%d')

    def count_annotations(self, dst_path):
        '''Counts the number of annotated structures stored in each text file.'''

        # start timer for metrics
        start = time.time()
        os.chdir(self.src_dir)

        # sets a file counter and a total annotations counter
        i = 0
        n = 0

        # makes directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # open write file to store data
        with open(f'{dst_path}/annotations.txt', 'w') as outfile:

            # open each annotations file to read the annotations
            for file in sorted(self.labels):
                with open(file, 'r') as readfile:
                    data = readfile.readlines()
                    total = len(data)

                # prints the number of annotations stored in each txt file
                print(f'There are n = {total} annotations in {file}.', file=outfile)

                # updates the total number of annotations and the file counter
                i = i + total
                n += 1

            # prints basic statistics and details to the outfile
            fin = time.time() - start
            print(f'There are n = {i} annotations across {n} the files checked.', file=outfile)

    def count_predictions(self, dst_path):
        '''Counts the number of predicted structures stored in each text file.'''

        # metrics on how the code is running
        start = time.time()
        os.chdir(self.src_dir)

        # sets a file counter and a total annotations counter
        i = 0
        n = 0

        # makes directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # open write file to store data
        with open(f'{dst_path}/predictions.txt', 'w') as outfile:

            # open each annotations file to read the annotations
            for file in sorted(self.labels):
                with open(file, 'r') as readfile:
                    data = readfile.readlines()
                    total = len(data)

                # prints the number of annotations stored in each txt file
                print(f'There are n = {total} predictions in {file}.', file=outfile)

                # updates the total number of annotations and the file counter
                i = i + total
                n += 1

            # prints basic statistics and details to the outfile
            fin = time.time() - start
            print(f'There are n = {i} predictions across {n} the files checked.', file=outfile)

    def count_image_dimensions(self, dst_path):
        '''records the dimensions of the image in the images in a given directory.'''

        # makes a directory to store the results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # makes a txt doc to store code details
        with open(f'{dst_path}/image_dimensions.txt', 'w') as outfile:

            # innit counter
            n = 0

            # loop over images
            for image in sorted(self.images):
                img = cv2.imread(f'{self.src_dir}/{image}', cv2.IMREAD_COLOR)

                # prints the shape of each image
                dims = img.shape[:2]

                print(f'{image} has dimensions {dims} (width x height)', file=outfile)

                # updates counter
                n += 1

            # prints some details and statistics to the outfile
            print(f'n = {n} file dimensions were checked.', file=outfile)

    def data_scatter_hist(self, dst_path):

        os.chdir(self.src_dir)

        # make the directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # innit lists to store bbox values
        x = []
        y = []
        h = []
        w = []

        # loop through the labels
        for label in self.labels:
            # read in the dirty yolo predictions file
            with open(f'{label}', 'r') as infile:
                for line in infile.readlines():
                    bbox = line.strip('\n').split(' ')

                    x.append(float(bbox[1]))
                    y.append(float(bbox[2]))
                    h.append(float(bbox[3]))
                    w.append(float(bbox[4]))

        s = 10
        figsz = (8, 8)
        dpi = 200
        fs = '14'
        ts = '18'

        plt.rcParams['font.size'] = fs

        fig = plt.figure(figsize=figsz, dpi=dpi)
        gs = fig.add_gridspec(3, 3)
        ax1_main = plt.subplot(gs[1:3, :2])
        ax1_xDist = plt.subplot(gs[0, :2], sharex=ax1_main)
        ax1_yDist = plt.subplot(gs[1:3, 2], sharey=ax1_main)

        ax1_main.scatter(x, y, marker='.', s=s)
        ax1_main.set(xlabel="X Data", ylabel="Y Data")

        ax1_xDist.hist(x, bins=100)
        ax1_xDist.set(ylabel='Counts')
        ax1_xDist.xaxis.set_label_position("top")
        ax1_xDist.xaxis.tick_top()

        ax1_yDist.hist(y, bins=100, orientation='horizontal')
        ax1_yDist.set(xlabel='Counts')
        ax1_yDist.yaxis.set_label_position("right")
        ax1_yDist.yaxis.tick_right()
        ax1_main.set_xlim(left=0, right=1)
        ax1_main.set_ylim(bottom=0, top=1)

        fig.suptitle('Bounding Box (x, y) Distribution')
        plt.savefig(f'{dst_path}/xy_distributions.png')
        plt.close(fig)

        fig = plt.figure(figsize=figsz, dpi=dpi)
        gs = fig.add_gridspec(3, 3)
        ax1_main = plt.subplot(gs[1:3, :2])
        ax1_xDist = plt.subplot(gs[0, :2], sharex=ax1_main)
        ax1_yDist = plt.subplot(gs[1:3, 2], sharey=ax1_main)

        ax1_main.scatter(h, w, marker='.', s=s)
        ax1_main.set(xlabel="Height Data", ylabel="Width Data")
        ax1_main.set_xlim(left=0, right=0.1)
        ax1_main.set_ylim(bottom=0, top=0.1)

        ax1_xDist.hist(h, bins=100)
        ax1_xDist.set(ylabel='Counts')
        ax1_xDist.xaxis.set_label_position("top")
        ax1_xDist.xaxis.tick_top()

        ax1_yDist.hist(w, bins=100, orientation='horizontal')
        ax1_yDist.set(xlabel='Counts')
        ax1_yDist.yaxis.set_label_position("right")
        ax1_yDist.yaxis.tick_right()

        fig.suptitle('Bounding Box (h, w) Distribution', fontsize=ts)
        plt.savefig(f'{dst_path}/hw_distributions.png')
        plt.close(fig)