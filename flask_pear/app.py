# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
import base64
import re
from os.path import exists
import time
from foodnet.food import outputimage,outputimage_test
app = Flask(__name__)
server ="192.168.1.126"

import pandas as pd
df=pd.read_csv('./static/class_result_nutrition.csv')

@app.route('/',methods = ['POST', 'GET'])
def upload():
   return render_template('uploadpic.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method=="POST":
      imgdata=request.form["myimg"]
      imgfile=request.form["imgfile"]
      imgdata=base64.b64decode(imgdata)
      with open("./static/images/"+imgfile+".png","wb") as f:
         f.write(imgdata)
      #output_file, item_class=outputimage(imgfile)
      
      output_file, item_class=outputimage_test(imgfile)
      item_message=df.iloc[0]
      
      path_to_file="static/images/"+imgfile+".png"
   return render_template("result.html",result=path_to_file,class_name=item_message['class_name'],
                          calories=item_message['calories'],fat=item_message['fat'],
                          carbs=item_message['carbs'],protein=item_message['protein'],
                          summary=item_message['summary'],more=item_message['more'],)

@app.route('/uploadpic1',methods = ['POST', 'GET'])
def upload1():
   return render_template('uploadpic 1.html')
@app.route('/uploadpic2',methods = ['POST', 'GET'])
def upload2():
   return render_template('uploadpic 2.html')
@app.route('/uploadpic3',methods = ['POST', 'GET'])
def upload3():
   return render_template('uploadpic 3.html')
@app.route('/uploadpic4',methods = ['POST', 'GET'])
def upload4():
   return render_template('uploadpic 4.html')
@app.route('/uploadpic5',methods = ['POST', 'GET'])
def upload5():
   return render_template('uploadpic 5.html')
if __name__ == '__main__':

   #from foodnet.inference import load_model
   
   #model = load_model('Y:/PEAR/train_logs/convnetXT_2022_Mar_19_AM_08_33_38/models/epoch_5.pth')
   app.run(debug = True,host=server,port=8000)
