#!/usr/bin/python

#from socket import gethostname
from imageai.Detection import ObjectDetection
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import os
from os import listdir
from os.path import isfile, join

execution_path = os.getcwd()

UPLOAD_FOLDER = execution_path+'/static'
RESULT_FOLDER = execution_path+'/static/result'
DATA_FOLDER = execution_path+'/static/data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config["DEBUG"] = True

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # processing
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=os.path.join(app.config['UPLOAD_FOLDER'], filename), output_image_path=os.path.join(app.config['RESULT_FOLDER'] , filename))
        detectionsTxt = ''
        horseCount = 0
        for eachObject in detections:
            if eachObject['name'] == 'horse':
                pointsStr = ''
                for points in  eachObject['box_points']:
                    pointsStr += str(points)+' '
                
                detectionsTxt += '<br>Percentage probability ' + str(eachObject['percentage_probability']) + '<br>'
                detectionsTxt += 'Box points ' + pointsStr + '<br>'
                detectionsTxt += '-----'
                horseCount = horseCount + 1

        returnHtml = 'Horses count ' + str(horseCount) + '<br>' + '-----' + detectionsTxt

        with open(os.path.join(app.config['DATA_FOLDER'], filename.rsplit('.', 1)[0].lower()) + '.txt', 'w') as f:
            f.write(str(returnHtml))
        return redirect('/')

@app.route('/', methods=['GET'])
def home():
    onlyfiles = [f for f in listdir(app.config['UPLOAD_FOLDER']) if isfile(join(app.config['UPLOAD_FOLDER'], f))]
    onlyTxt = [f for f in listdir(app.config['DATA_FOLDER']) if isfile(join(app.config['DATA_FOLDER'], f))]

    data = {}
    for txt in onlyTxt:
        f=open(os.path.join(DATA_FOLDER, txt), "r")
        contents =f.read()
        data[txt] = contents
    
    return render_template('index.html',files = onlyfiles, data = data)

if __name__=="__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'), 
            port=int(os.getenv('PORT', 8007)))