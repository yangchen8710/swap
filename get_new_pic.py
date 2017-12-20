from PIL import Image,ImageDraw
import imageio
import cv2
import numpy as np
import face_recognition
import json
import sys 
sys.path.append('f_swap')
from pathlib import Path

from utils import get_image_paths

from model import autoencoder_A
from model import autoencoder_B
from model import encoder, decoder_A, decoder_B
encoder  .load_weights( "f_swap/models/encoder.h5"   )
decoder_A.load_weights( "f_swap/models/decoder_A.h5" )
decoder_B.load_weights( "f_swap/models/decoder_B.h5" )

filename = "short_hamilton_clip.mp4"
reader = imageio.get_reader(filename,  'ffmpeg')
fps = reader.get_meta_data()['fps']

def get_center(top, right, bottom, left):
    center_x = (int)((right+left)/2)
    center_y = (int)((top+bottom)/2)
    return center_x,center_y

#160x160 to 256x256
def get_m(top, right, bottom, left):
    height = bottom - top
    width = right -left
    print(str(height)+":"+str(width))
    std = 0
    if height>width:
        #std = (int)(1.0 * height / 2  / 160 * 256)
        std = (int)(1.0 * height /2)
    else:
        #std = (int)(1.0 * width / 2  / 160 * 256)
        std = (int)(1.0 * width / 2)
    center_x, center_y = get_center(top, right, bottom, left)
    return center_y - std, center_x + std, center_y + std, center_x - std  

def convert_one_face( autoencoder, image ):
    origin_shape = image.shape
    print(origin_shape)
    face = cv2.resize( image, (64,64) )
    face = np.expand_dims( face, 0 )
    new_face = autoencoder.predict( face / 255.0 )[0]
    new_face = np.clip( new_face * 255, 0, 255 ).astype( image.dtype )
    new_face = cv2.resize( new_face, (origin_shape[0],origin_shape[0]) )
    return new_face

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
    array = np.array(pil_image)    
    opencvImage = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    copyCVImage = opencvImage.copy()

    for i,rect in enumerate(rect_list):
        rectp = rect.get("rect")
        top, right, bottom, left = rectp[0],rectp[1],rectp[2],rectp[3]
        top, right, bottom, left = get_m(top, right, bottom, left)
        
        
        face_image = opencvImage[top:bottom, left:right]
        new_face = convert_one_face(autoencoder_B,face_image)
        center_x, center_y = get_center(top, right, bottom, left)
        x_offset=center_x - (int)(new_face.shape[0]/2) 
        y_offset=center_y - (int)(new_face.shape[0]/2)
        copyCVImage[y_offset:y_offset + new_face.shape[0], x_offset:x_offset +new_face.shape[1]] = new_face
    cv2.imshow("full",copyCVImage)
    cv2.waitKey((int)(1000/1000))
        #draw.rectangle([left,top,right,bottom])
    #array = np.array(pil_image)
    #opencv_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    #sp = opencv_image.shape
    #width_cv = (int)(sp[1]/2)
    #height_cv = (int)(sp[0]/2)
    #opencv_image = cv2.resize(opencv_image,(width_cv,height_cv))
    #cv2.imshow("full",opencv_image)
    #cv2.waitKey((int)(1000/10))
