# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:30:46 2024

@author: User
"""

import sys
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

def pprint_image(image, title = 'Надпись'):
    gs = plt.GridSpec(2, 2)
    plt.figure(figsize=(32, 40))
    plt.subplot(gs[0])
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.show()

#plt.rcParams["figure.figsize"] = [6, 4]
image1 = cv.imread('./chinese.jpg')
image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
pprint_image(image1, 'Исходное')
ret, thresh = cv.threshold(image1, 120, 255, cv.THRESH_BINARY)
ret, bad_binary = cv.threshold(image1, 200, 255, cv.THRESH_BINARY)
pprint_image(thresh, title = 'Бинаризация (без линий)')
pprint_image(bad_binary, title = 'Бинаризация (с линиями)')
adapt = cv.adaptiveThreshold(image1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
pprint_image(adapt, title = 'Бинаризация адаптивная')
resid = cv.subtract(adapt, bad_binary)
pprint_image(resid, title = 'Разность адаптивной и прямой бинаризаций')
median_resid = cv.medianBlur(resid, 3)
pprint_image(median_resid, title = 'Медианный фильтр на разность изображений')
minus_thresh = cv.bitwise_not(thresh)
good_bin = cv.add(resid, minus_thresh)
good_bin = cv.bitwise_not(good_bin)
pprint_image(good_bin, title = 'Попытка восстановить содержание линий')