import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
global model, graph
import tensorflow as tf
graph = tf.compat.v1.get_default_graph()
import re
import nltk
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


model = load_model('mymodel.h5') 
app = Flask(__name__) 
@app.route('/') 
def home():
    return render_template('index.html')
@app.route('/prediction', methods = ['GET','POST'])
def pred():
    if request.method == 'POST':
        review = request.form['message']
        review = re.sub('[^a-zA-Z]', ' ',review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word 
                  in set(stopwords.words('english'))]
        review = ' '.join(review)
        
        with graph.as_default():
            y_p = model.predict(review)
        if y_p.argmax() == 0: 
            output = "Average"
        elif y_p.argmax() == 1:
            output = "Good"
        else:
            output = "Poor"
        return render_template('index.html',prediction = 
                               ("The Customer review is " + output)) 
    else:
        return render_template('index.html')
            

if __name__ == "__main__":
    app.run(debug = True)
    
                