import time
import cv2

def outputimage(imgfile):
   #input_file="./static/images/"+imgfile+".png"
   input_file="./static/img/test3.jpg"
   output_file="./static/images/"+imgfile+"_result.png"
   path_to_txt="./static/images/"+imgfile+"_result.txt"
   img = cv2.imread(input_file)
   cv2.imwrite(output_file, img)
   with open(path_to_txt, 'w') as outfile:
      outfile.write('I am a gan guo bao cai')
   time.sleep(5)
   return output_file

