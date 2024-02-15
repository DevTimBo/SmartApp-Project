# Pipeline to recognize handwritten entries done in the "Bafoeg" form

A project done by the students of the HS-Bremen in the context of an module called "Smartapp"

## Requirements:

you need a working python environment and some pip installations like [Tensorflow](https://www.tensorflow.org/install) and [openCV](https://keras.io/keras_cv/#keras-3-installation)

IMPORTANT: first install openCV and then Tensorflow you could get a version conflict otherwise. 

TIP: OpenCV-v2 is at the moment enough to run this project. 

## Run the Pipeline/App

### There are two options to run the Pipeline:

#### 1. Pipeline:

You can run the Pipeline by executing "pipeline.py" with your favorite python installation.

Pro-Tip: You can also specify parameters. you get a help how to do this with "pipeline.py --help"

Noch nicht implementiert ???

#### 2. App (Android):

If you go into the folder "tfliteapp" you will find an flutter Repository with it's own ReadMe you will find further instructions there. 

#### 3. API: 

The API_SmartAPP.py serves as an interface for the pipeline. The input can be found in the "API" folder under "images". The input is saved as "page.jpg". To test the API as a single module, it is recommended to use Postman.