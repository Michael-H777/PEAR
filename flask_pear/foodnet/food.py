import time
import cv2

from foodnet.inference import forward 

def outputimage(imgfile): # image 
   input_file="./static/images/"+imgfile+".png"
   # input_file="./static/img/test3.jpg"
   output_file="./static/images/"+imgfile+".png"
   path_to_txt="./static/images/"+imgfile+"_result.txt"
   img = cv2.imread(input_file)
   
   result = forward(model, img)
   
   return output_file, result

