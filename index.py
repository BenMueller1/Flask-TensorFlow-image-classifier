from flask import Flask, render_template
from flask_restful import Api, Resource
# import the model here

app = Flask(__name__)
api = Api(app)


@app.route("/")
def index():
    return render_template("index.html")


# add an api endpoint, this is where submitted image is sent
class ImageClassificationEndpoint(Resource):
    def post(self):
        # TODO, use the model we have trained to clasify the data
        # return a JSON object that contains the classification results
        pass

api.add_resource(ImageClassificationEndpoint, "/classifyImage")



if __name__ == "__main__":
    app.run(debug=True)