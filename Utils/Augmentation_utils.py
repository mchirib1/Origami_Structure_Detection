import os
import random
import shutil
import time
from datetime import date

import cv2
import numpy as np


# convert from Yolo_mark to opencv format
def yoloFormattocv(x1, y1, x2, y2, H, W):
    bbox_width = x2 * W
    bbox_height = y2 * H
    center_x = x1 * W
    center_y = y1 * H

    voc = []

    voc.append(center_x - (bbox_width / 2))
    voc.append(center_y - (bbox_height / 2))
    voc.append(center_x + (bbox_width / 2))
    voc.append(center_y + (bbox_height / 2))

    return [int(v) for v in voc]


# convert from opencv format to yolo format
# H,W is the image height and width
def cvFormattoYolo(corner, H, W):
    bbox_W = corner[3] - corner[1]
    bbox_H = corner[4] - corner[2]

    center_bbox_x = (corner[1] + corner[3]) / 2
    center_bbox_y = (corner[2] + corner[4]) / 2

    return corner[0], float(center_bbox_x / W), float(center_bbox_y / H), float(bbox_W / W), float(bbox_H / H)


class YoloRotatebbox:
    '''
    Citation: https://github.com/usmanr149/Yolo_bbox_manipulation
    '''

    def __init__(self, filename, image_ext, angle):
        self.filename = filename
        self.image_ext = image_ext
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext, 1)

        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotateYolobbox(self):
        # shape has 3 dimensions for color images
        new_height, new_width = self.rotate_image().shape[:2]

        # reads in the current annotations
        f = open(self.filename + '.txt', 'r')

        # stores all the lines
        f1 = f.readlines()

        new_bbox = []

        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')  # stores each of the values seperately in a list

            # yolo format is [class, center x, center y, bbox height, bbox width]
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)

                # calculates the upper left and upper right hand corners
                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):

                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                        # print(x_prime, y_prime)

                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                 new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat


class DataAugmenter:
    '''A class to help with data augmentation of the triangle image analysis data.  Handles things like numerical
    augmentations as well as rotations.'''

    def __init__(self, src_dir):
        self.src_dir = src_dir

        self.images = [file for file in os.listdir(src_dir) if
                       file.split(".")[1] == 'png' or file.split(".")[1] == 'jpg']
        self.labels = [file for file in os.listdir(src_dir) if file.split(".")[1] == 'txt'
                       and file != 'classes.txt']

        self.date = date.today().strftime('%Y%m%d')

    def resizing(self, n, dst_path):
        '''Rescales all the images in a directory to a random height and width'''

        os.chdir(self.src_dir)

        # make directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        for _ in range(0, n):

            for image in sorted(self.images):
                image_name = image.split('.')[0]
                # read in image
                img = cv2.imread(image, cv2.IMREAD_COLOR)

                # desired height and width
                h, w = img.shape[:2]

                r = round(random.uniform(0.65, 1),2)

                new_h = round(h * r)
                new_w = round(w * r)

                resized = cv2.resize(img,(new_w, new_h))

                pad = np.zeros(img.shape[:3])
                y_offset = 0
                x_offset = 0

                pad[x_offset:resized.shape[0] + x_offset, y_offset:resized.shape[1] + y_offset] = resized

                cv2.imwrite(f'{dst_path}/{image_name}_scaled.png', pad)

        for label in self.labels:
            label_name = label.split('.')[0]
            shutil.copy(label, f'{dst_path}/{label_name}_scaled.txt')

    def cvt_grayscale(self, dst_path):
        '''Converts all the images in a directory to grayscale'''

        # make directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # innit counter
        n = 0

        # loop through the images
        for image in self.images:
            image_name = image.split('.')[0]
            img_gray = cv2.imread(f'{self.src_dir}/{image}', cv2.IMREAD_GRAYSCALE)

            cv2.imwrite(f'{dst_path}/gray_{image_name}.png', img_gray)

            # update counter
            n += 1

        # also copies the annotations to the results directory
        for file in self.labels:
            shutil.copy(f'{self.src_dir}/{file}', f'{dst_path}/gray_{file}')

        print(f'{n} images converted to grayscale.')

    def cvt_alpha_beta(self, alpha, beta, dst_path):
        '''Increases the contrast of all images in a directory by controlling the alpha and beta parameters

        One can consider f(i,j) the source pixel and g(i,j) the edited pixel intensity. Controlling the alpha and beta
        values modulate pixel intensity according to the equation:       g(i,j) = alpha * f(i,j) + beta '''

        os.chdir(self.src_dir)

        a = alpha  # scales contrast
        b = beta  # scales brightness

        # makes a directory to store the results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # innit counter
        n = 0

        # loop through images
        for image in self.images:
            image_name = image.split('.')[0]
            img = cv2.imread(f'{self.src_dir}/{image}', cv2.IMREAD_COLOR)

            #  utilize opencv method instead of "triple for loop"
            new_image = cv2.convertScaleAbs(img, alpha=a, beta=b)

            # simple logic to avoid decimal file names
            # to avoid decimals in file names an alpha < 0 are named 'lowcon'
            # a better naming convention is needed
            if a < 1:
                cv2.imwrite(f'{dst_path}/alphaLocon_beta{b}_{image_name}.png', new_image)
            else:
                cv2.imwrite(f'{dst_path}/alpha{int(a)}_beta{b}_{image_name}.png', new_image)

            # update counter
            n += 1

        # copies the annotations into the results directory with the same name as the edited images
        for file in self.labels:
            if a < 1:
                shutil.copy(f'{self.src_dir}/{file}', f'{dst_path}/alphaLocon_beta{b}_{file}')
            else:
                shutil.copy(f'{self.src_dir}/{file}', f'{dst_path}/alpha{int(a)}_beta{b}_{file}')

        print(f'{n} images scaled using alpha={a} and beta={b}.')

    def rotate_img(self, n_rotations, r_angle, dst_path):
        '''Rotates an image n times with the  angle of rotation randomly selected from 1 < r < angle.'''

        os.chdir(self.src_dir)

        # makes a directory to store results in
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # sets the number of rotations which need to be augmented
        for i in range(0, n_rotations):

            # loops over each image to rotate
            for image in self.images:
                image_name = image.split('.')[0]

                # sets random integer for rotation angle
                ang = random.randint(1, r_angle)

                # sets the data into the rotated image class and then rotates the image
                data = YoloRotatebbox(image_name, '.png', ang)
                img_rotated = data.rotate_image()

                # saves the rotated image in the results directory
                cv2.imwrite(f'{dst_path}/{image_name}_r{ang}.png', img_rotated)

                # opens a new text doc to write in new bbox coordinates
                with open(f'{dst_path}/{image_name}_r{ang}.txt', 'w') as outfile:

                    # rotates each bbox
                    for line in data.rotateYolobbox():
                        H, W = img_rotated.shape[:2]
                        line2 = cvFormattoYolo(line, H, W)

                        # stores the new bbox coordinates in the txt doc
                        print(f'{line2[0]} {line2[1]:0.6f} {line2[2]:0.6f} {line2[3]:0.6f} {line2[4]:0.6f}',
                              file=outfile)

        print(f'{len(self.images)} images rotated {n_rotations} times. Output:{len(self.images)*n_rotations} images')

    def quadrant_1(self, dst_path):

        os.chdir(self.src_dir)

        # make the directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # open txt doc to write out statistics
        with open(f'{dst_path}results.txt', 'w') as outfile:
            # counter
            n = 0

            # loop over images in the src directory
            for image in self.images:
                image_name = image.split('.')[0]
                img = cv2.imread(image, cv2.IMREAD_COLOR)

                rows = img.shape[0]
                cols = img.shape[1]

                half_rows = int(round((rows / 2), 0))
                half_cols = int(round((cols / 2), 0))

                img[half_rows:, :, :] = 0  # send rows to 0 for black, 255 for white
                img[:, half_cols:, :] = 0  # send columns to 0 for black, 255 for white

                # writes new images to the results directory
                cv2.imwrite(f'{dst_path}/1quad_{image_name}.png', img)

                # updates counter
                n += 1

            # writes out some performance values
            print(f'n = {n} images were resized changed to quadrant 1 only.', file=outfile)

    def quadrant_3(self, dst_path):

        os.chdir(self.src_dir)

        # make the directory to store results
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass

        # open txt doc to write out statistics
        with open(f'{dst_path}/results.txt', 'w') as outfile:
            # counter
            n = 0

            for image in self.images:
                image_name = image.split('.')[0]
                img = cv2.imread(image, cv2.IMREAD_COLOR)

                rows = img.shape[0]
                cols = img.shape[1]

                half_rows = int(round((rows / 2), 0))
                half_cols = int(round((cols / 2), 0))

                img[:half_rows, :, :] = 0  # send rows to 0 for black, 255 for white
                img[:, :half_cols, :] = 0  # send columns to 0 for black, 255 for white

                # writes new images to the results directory
                cv2.imwrite(f'{dst_path}/3quad_{image_name}.png', img)

                # updates counter
                n += 1

            # writes out some performance values
            print(f'n = {n} images were resized changed to quadrant 3 only.', file=outfile)