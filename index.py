from flask import Flask, render_template, redirect, url_for, flash, request
from flask_restful import Api, Resource
from flask import jsonify

from PIL import Image
import os
# import the model here

app = Flask(__name__)
api = Api(app)

# UPLOAD_FOLDER = '/uploaded_images'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    # maybe add code to get some info about the model here? 
    # So that I don't have to edit the html file if I decide to change the model
    return render_template("about.html")


@app.route("/ben")
def ben():
    return render_template("ben.html")


@app.route("/classifyImage", methods=['POST'])
def classify():
    # TODO create model, then use it to classify image
    img_file = request.files['image']
    img = Image.open(img_file)
    # img.show()
    prediction="no clue :("
    certainty=100
    return render_template("prediction.html", prediction=prediction, certainty=certainty)




if __name__ == "__main__":
    app.run(debug=True)