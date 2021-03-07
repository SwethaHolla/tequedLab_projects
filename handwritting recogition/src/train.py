#prg is to teach the alg to learn to recognise the nos. ,with the help of response, from the img.  
#!/usr/bin/python               .this prg is used to read a img with no. in it. then detects the an area of 50 and puts a rectangle around it.
# -*- coding: utf-8 -*-         response is feed to the alg and then it moves on to the next,img of 50 area, then repeats the procces and stored the following
#1)sample- image data 2)keyboard keys pressed -response
__author__ = 'yan9yu'

import sys
import numpy as np
import cv2

im = cv2.imread('../data/handwritten.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2) #

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#chainapprox is used to compress mathematicaly

samples = np.empty((0, 100), np.float32)#create empty arrays (0,100)gives the the size and float gives 32 bit data and type
responses = []
keys = [i for i in range(65, 90)] #makes the algoritm to read from 1-10

for cnt in contours:

    if cv2.contourArea(cnt) > 50:#countour array is greater than 50 is detected(i.e it detects a no.(img) whose size is greater than 50 like area around the no.) 
        [x, y, w, h] = cv2.boundingRect(cnt) #thedata i.e the height,width,x coord and ycord is stored

        if h > 28:

            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 10))#we are taking the img(the small no..) and then storing it in roismall, we also reduce the size for storing 
            cv2.imshow('norm', im) #to get the reponse from the user 
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int((key)))
                sample = roismall.reshape((1, 100))#the img is reshaped,and compare the img area and checks if the size is smaller than 1-100
                samples = np.append(samples, sample, 0)#0?

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))

print("training complete")
samples = np.float32(samples)
responses = np.float32(responses)
cv2.imwrite("../data/train_result.png", im)#used to display on img
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)
