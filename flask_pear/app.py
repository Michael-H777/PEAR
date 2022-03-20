# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
import base64
import re
from os.path import exists
import time
from foodnet.food import outputimage,outputimage_test
app = Flask(__name__)
server ="192.168.1.126"


@app.route('/',methods = ['POST', 'GET'])
def upload():
   return render_template('uploadpic.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method=="POST":
      imgdata=request.form["myimg"]
      imgfile=request.form["imgfile"]
      print(imgdata)
      imgdata=base64.b64decode(imgdata)
      with open("./static/images/"+imgfile+".png","wb") as f:
         f.write(imgdata)
      #output_file, item_class=outputimage(imgfile)
      
      output_file, item_class=outputimage_test(imgfile)
      
      print(item_class)
      path_to_file="static/images/"+imgfile+".png"
   return render_template("result.html",result=path_to_file,message=item_class)
         
if __name__ == '__main__':

   #from foodnet.inference import load_model
   
   #model = load_model('Y:/PEAR/train_logs/convnetXT_2022_Mar_19_AM_08_33_38/models/epoch_5.pth')
   app.run(debug = True,host=server,port=8000)
