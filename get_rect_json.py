from PIL import Image,ImageDraw
import imageio
import cv2
import numpy as np
import face_recognition
import json
import time
filename = "01.bin"

reader = imageio.get_reader(filename,  'ffmpeg')
fps = reader.get_meta_data()['fps']



count = 0
range_1 = count
range_2 = count + 10000
json_name = str(range_1)+"_"+str(range_2)+".json"
fw = open(json_name,'w')
#for im in reader:

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
    shape1 = opencv_image.shape
    opencv_image = cv2.resize( opencv_image, ((int)(shape1[1]/2),(int)(shape1[0]/2)) )
    face_locations = face_recognition.face_locations(opencv_image, number_of_times_to_upsample=2, model="cnn")
    str_res = ("{}".format(len(face_locations)))
    str_res = "Frame"+ str(count) + ": "  + str_res  
    print(str_res)
    res = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        res.append({"rect":(top, right, bottom, left)})
    resall.append({"rect_list":res,"count":count})
    count = count + 1
    time_end=time.time()
    print(time_end-time_start)

jsonres = json.dump(resall,fw)
fw.close()
