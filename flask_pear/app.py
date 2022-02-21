from flask import Flask, render_template, request, redirect, url_for
import base64
app = Flask(__name__)
server ="192.168.1.126"

@app.route('/',methods = ['POST', 'GET'])
def upload():
   if request.method=="POST":
      imgdata=request.form["myimg"]
      imgdata=base64.b64decode(imgdata)
      with open("photo.png","wb") as f:
         f.write(imgdata)
   return render_template('uploadpic.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   
   return render_template("result.html")

if __name__ == '__main__':
   app.run(debug = True,host=server,port=8000)
