# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 19:09:45 2019

@author: DG
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:42:08 2019

@author: DG
"""
#import Libraries
import cv2
import numpy as np

#Path for Label list and read it in labels
labels_path = 'coco.names'
labels = open(labels_path).read().split("\n")

#define confidence level and threshold
pre_conf = 0.5
threshold = 0.3

#Define different random colors for boxes
np.random.seed(0)
colors = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

#path for YOLO_wightss and config file
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'

#Define neural net
net = cv2.dnn.readNetFromDarknet(config_path,weights_path)
ln = net.getLayerNames()

ln =[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]
camera = cv2.VideoCapture("http://192.168.43.1:8080/video")
cv2.namedWindow('name',cv2.WINDOW_NORMAL)
while True:
    ret,img = camera.read()    
    #read image for detect objects and find height,width
    img = cv2.resize(img,(416,416))
    (H,W) = img.shape[:2]
    
    #Find YOLO Output layers 
    #give image as input to neural network
    blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
    net.setInput(blob)
    out_lay = net.forward(ln)


    
           #Identify best prediction and draw bounding boxes
    for output in out_lay:
        
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence>pre_conf:
                box = detection[0:4]*np.array([W,H,W,H])
                (cenX,cenY,w,h) = box.astype('int32')
                
                x = int(cenX-(w/2))
                y = int(cenY-(h/2))
                color = [int(c) for c in colors[classId]]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                text = print("{}:{:.4f}".format(labels[classId],confidence))
                cv2.putText(img,text,(cenX,cenY),cv2.FONT_HERSHEY_DUPLEX,2,color,4)


    
        cv2.imshow("name",cv2.resize(img,(500,500)))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

camera.close()
cv2.destroyAllWindows()
    
    
