from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      impath = request.form['impath']
      return render_template("result.html",result = './static/img/test3.jpg',message = 'dry pot bag veg')

if __name__ == '__main__':
   app.run(debug = True)
