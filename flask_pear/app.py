from flask import Flask, render_template, request, redirect, url_for
import base64
from os.path import exists
import time
app = Flask(__name__)
server ="192.168.1.126"

@app.route('/',methods = ['POST', 'GET'])
def upload():
   if request.method=="POST":
      imgdata=request.form["myimg"]
      imgfile=request.form["imgfile"]
      imgdata=base64.b64decode(imgdata)
      print(imgfile)
      with open("./static/images/"+imgfile+".png","wb") as f:
         f.write(imgdata)
   return render_template('uploadpic.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method=="POST":
      imgfile=request.form["imgfile"]
      path_to_file="static/images/"+imgfile+".png"
      for i in range(10):
         if exists(path_to_file):
            return render_template("result.html",result=path_to_file,message="idk")
         else:
            time.sleep(1)
   return render_template("result.html",result="static/img/test3.png",message="dry pot bag veg")  
      
if __name__ == '__main__':
   app.run(debug = True,host=server,port=8000)
