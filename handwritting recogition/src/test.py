#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import cv2
import numpy as np

#######   training part    ###############
samples = np.loadtxt('E:\projects\Tequed Lab\handwritting recogition\srcgeneralsamples.data', np.float32)#extraction of ample and response
responses = np.loadtxt('E:\projects\Tequed Lab\handwritting recogition\srcgeneralresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)#


############################# testing part  #########################

im = cv2.imread('../data/handwritten.png')#read img
out = np.zeros(im.shape, np.uint8)#white img is converted to zeros,8bit unsigned integers       <---
print(im.shape)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)#color conversion
thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)#11 nearset neighbours

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 50:#trying to recognise no. or alpha
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))
            roismall = roismall.reshape((1, 100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)#result will have key responses
            string = str(int((results[0][0])))#result is converted to array
            cv2.putText(out,chr(int(string)), (x, y + h), 0, 1, (0, 255, 0))

cv2.imshow('im', im)#displays in and out
cv2.imshow('out', out)
cv2.waitKey(0)
