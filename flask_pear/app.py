from flask import Flask, render_template, request, redirect, url_for
import base64
from os.path import exists
import time
from foodnet.food import outputimage
app = Flask(__name__)
server ="192.168.1.4"


@app.route('/',methods = ['POST', 'GET'])
def upload():
   if request.method=="POST":
      imgdata=request.form["myimg"]
      imgfile=request.form["imgfile"]
      imgdata=base64.b64decode(imgdata)
      with open("./static/images/"+imgfile+".png","wb") as f:
         f.write(imgdata)
      output_file, item_class=outputimage(imgfile)
      
      print(item_class)
   return render_template('uploadpic.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method=="POST":
      imgfile=request.form["imgfile"]
      path_to_file="static/images/"+imgfile+"_result.png"
      path_to_txt="static/images/"+imgfile+"_result.txt"
      for i in range(10):
         if exists(path_to_file) and exists(path_to_txt):
            with open(path_to_txt, "r") as f:
               txtresult=f.read()
            return render_template("result.html",result=path_to_file,message=txtresult)
         else:
            time.sleep(1)
   return render_template("result.html",result="static/img/oops.jpg",message="Too Long Too wait")  
      
if __name__ == '__main__':

   from foodnet.inference import load_model
   
   model = load_model('Y:/PEAR/train_logs/convnetXT_2022_Mar_19_AM_08_33_38/models/epoch_5.pth')
   app.run(debug = True,host=server,port=8000)
