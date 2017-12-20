from PIL import Image,ImageDraw
import imageio
import cv2
import numpy as np
import face_recognition
import json
filename = "short_hamilton_clip.mp4"
reader = imageio.get_reader(filename,  'ffmpeg')
fps = reader.get_meta_data()['fps']

def get_m(face_location):
    top, right, bottom, left = face_location
    height = bottom - top
    width = right -left
    std = (int)(1.0 * width / 2  / 160 * 256)
    if height>width:
        std = (int)(1.0 * height / 2  / 160 * 256)
    center_x = (int)((right+left)/2)
    center_y = (int)((top+bottom)/2)
    return center_y - std, center_x + std, center_y + std, center_x - std   

fw = open('myu_s.json','w')
resall = []
nums = [120]
count = 200
#for im in reader:
for x in range(200,250):
    im = reader.get_data(count)
    pil_image = Image.fromarray(im)
    #pil_image.show()
    #print(num)
    #cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    #cv2.imshow("full",open_cv_image)
    pil_image.save('temp.jpg', 'jpeg')
    image = face_recognition.load_image_file("temp.jpg")
    #face_landmarks_list = face_recognition.face_landmarks(image)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    #print(num)
    print("I found {} face(s).".format(len(face_locations)))
    res = []
    for face_location in face_locations:
        draw = ImageDraw.Draw(pil_image)
        top, right, bottom, left = face_location
        #top, right, bottom, left = get_m(face_location)
        res.append({"rect":(top, right, bottom, left)})

        #draw.line((left,top) +(right,bottom), fill=128)
        #draw.rectangle([left,top,right,bottom ])

        # Print the location of each face in dr image
        #top, right, bottom, left = face_location
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        array = np.array(pil_image)    
        opencvImage = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        opencvImage = cv2.resize( opencvImage, (256,256) )
        cv2.imwrite("temp1.png",opencvImage)
        #open_cv_image = np.array(pil_image.convert('RGB'))  
        #cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        #cv2.imshow("temp1",open_cv_image)
    #array = np.array(pil_image)    
    #opencvImage = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    resall.append({"rect_list":res,"count":count})
    count = count + 1
jsonres = json.dump(resall,fw)
fw.close()
#fr = open('myu_s.json','r')
    #writer.append_data(array)
    #open_cv_image = np.array(pil_image.convert('RGB')) 
    #cv2.imshow("full",opencvImage)
    #cv2.waitKey()
#writer.close()

