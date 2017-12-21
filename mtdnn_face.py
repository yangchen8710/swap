# coding=utf-8
"""Face Detection and Recognition"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# This is the work of David Sandberg and shanren7 remodelled into a
# high level container. It's an attempt to simplify the use of such
# technology and provide an easy to use facial recognition package.
#
# https://github.com/davidsandberg/facenet
# https://github.com/shanren7/real_time_face_recognition
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
import os
from PIL import Image,ImageDraw
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc


#import facenet
import sys
sys.path.append("align/")
import face
detector = face.Detection()

import imageio
filename = "01.bin"
reader = imageio.get_reader(filename,  'ffmpeg')
fps = reader.get_meta_data()['fps']

import time

count = 0
range_1 = count
range_2 = count + 10000
json_name = str(range_1)+"_"+str(range_2)+".json"
fw = open(json_name,'w')

resall = []
for x in range(range_1,range_2):
    if(count%10!=0):
        count = count + 1
        continue
    time_start=time.time()
    im = reader.get_data(count)
    pil_image = Image.fromarray(im)
    array = np.array(pil_image)    
    opencv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    face_locations = detector.find_faces(opencv_image)
    str_res = ("{}".format(len(face_locations)))
    str_res = "Frame"+ str(count) + ": "  + str_res  
    print(str_res)
    res = []
    for face_location in face_locations:
        box = face_location.bounding_box
        a = box[0]
        b = box[1]
        c = box[2]
        d = box[3]
        res.append({"rect":(a,b,c,d)})
    resall.append({"rect_list":res,"count":count})
    count = count + 1
    time_end=time.time()
    print(time_end-time_start)

jsonres = json.dump(resall,fw)
fw.close()



    #cropped = image[face1.bounding_box[1]:face1.bounding_box[3], face1.bounding_box[0]:face1.bounding_box[2], :]
    #cv2.imshow('1.png',cropped)
    #cv2.waitKey(0)