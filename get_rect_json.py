from PIL import Image,ImageDraw
import imageio
import cv2
import numpy as np
import face_recognition
import json
filename = "short_hamilton_clip.mp4"
fw = open('myu_s.json','w')

reader = imageio.get_reader(filename,  'ffmpeg')
fps = reader.get_meta_data()['fps']

resall = []
nums = [120]
count = 30
range_1 = count
range_2 = count + 20
#for im in reader:
for x in range(range_1,range_2):
    im = reader.get_data(count)
    pil_image = Image.fromarray(im)
    array = np.array(pil_image)    
    opencv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
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

jsonres = json.dump(resall,fw)
fw.close()
