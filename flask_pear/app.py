from flask import Flask, render_template, request
app = Flask(__name__)
server ="192.168.1.126"

@app.route('/')
def student():
   return render_template('index.html',ip="/result")

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      impath = request.form['impath']
      
      return render_template("result.html",result = './static/img/test3.jpg',message = 'dry pot bag veg')

if __name__ == '__main__':
   app.run(debug = True,host=server,port=8000)
