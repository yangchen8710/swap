from PIL import Image,ImageDraw
import imageio
import cv2
import numpy as np
import face_recognition
import json
filename = "short_hamilton_clip.mp4"
reader = imageio.get_reader(filename,  'ffmpeg')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer("short_hamilton_clip_1.mp4", fps=fps)

#160x160 to 256x256
def get_m(top, right, bottom, left):
    height = bottom - top
    width = right -left
    print(str(height)+":"+str(width))
    std = 0
    if height>width:
        std = (int)(1.0 * height / 2  / 160 * 256)
        #std = (int)(1.0 * height /2)
    else:
        std = (int)(1.0 * width / 2  / 160 * 256)
        #std = (int)(1.0 * width / 2)
    center_x = (int)((right+left)/2)
    center_y = (int)((top+bottom)/2)
    return center_y - std, center_x + std, center_y + std, center_x - std   

def cut(pil_image,top, right, bottom, left,frame,index):
    try:
        array = np.array(pil_image)    
        opencvImage = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        
        face_image = opencvImage[top:bottom, left:right]
        face_image = cv2.resize( face_image, (256,256) )
        cv2.imwrite("./1/"+str(frame)+"_"+str(index)+'.png',face_image)
        return True
    except:
        print("out1")
        return False
    else:
        print("out2")
        return False

fr = open('myu_s.json','r')
readres = json.load(fr)
countpic = 0
for frame in readres:
    rect_list = frame.get("rect_list")
    count = frame.get("count")

    print("Frame" + str(count)+":"+str(len(rect_list)))

    im = reader.get_data(count)
    pil_image = Image.fromarray(im)
    draw = ImageDraw.Draw(pil_image)
    for i,rect in enumerate(rect_list):
        rectp = rect.get("rect")
        top, right, bottom, left = rectp[0],rectp[1],rectp[2],rectp[3]
        top, right, bottom, left = get_m(top, right, bottom, left)
        if(cut(pil_image,top, right, bottom, left,count,i)):
            countpic = countpic + 1
        #draw.rectangle([left,top,right,bottom])
    #array = np.array(pil_image)
    #opencv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    #sp = opencv_image.shape
    #width_cv = (int)(sp[1]/2)
    #height_cv = (int)(sp[0]/2)
    #opencv_image = cv2.resize(opencv_image,(width_cv,height_cv))
    #cv2.imshow("full",opencv_image)
    #cv2.waitKey((int)(1000/10))

