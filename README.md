# Pipeline to recognize handwritten entries done in the "Bafoeg" form

A project done by the students of the HS-Bremen in the context of an module called "Smartapp"

## Requirements:

you need a working python environment and some pip installations like [Tensorflow](https://www.tensorflow.org/install) and [openCV](https://keras.io/keras_cv/#keras-3-installation)

IMPORTANT: first install openCV and then Tensorflow you could get a version conflict otherwise. 

TIP: OpenCV-v2 is at the moment enough to run this project. 

## Run the Pipeline/App

### There are three options to run the Pipeline:

#### 1. Inferenz Pipeline Notebook:

- You have to pip install the requirements.txt file and then you can run the notebook. 
- The notebook is called "inferenz_Pipeline.ipynb"
#### 2. App (Android):

If you go into the folder "tfliteapp" you will find an flutter Repository with it's own ReadMe you will find further instructions there. 

#### 3. API: 

The API_SmartAPP.py serves as an interface for the pipeline. The input can be found in the "API" folder under "images". The input is saved as "page.jpg". To test the API as a single module, it is recommended to use Postman.

#### 4. App (Raspberry Pi):
In the folder "Raspberry_Pi," you will find additional information along with its own ReadMe file.

### How to use the Handwriting Training Notebooks
#### IAM Training Notebook:
- The IAM Training Notebook is used to train the model on the IAM dataset.
- handwriting_training.ipynb is the notebook to train the model on the IAM dataset.
- The IAM Dataset needs to be downloaded from the official website and the path to the dataset needs to be changed in the notebook.
- https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- You will need a account to download the dataset.
- We only use the lines part of the dataset.
#### Transfer Training Notebook:
- The Transfer Training Notebook is used to train the model our Bafoeg dataset containing the 1st page
- transfer_learning.ipynb is the notebook to train the model on the Bafoeg dataset.
- the dataset is in data_zettel/filled_resized (images) and data_zettel/Annotations (xml files with the annotations)
- with this you can create a dataset for the transfer learning notebook with the help of the "dataset_creator.ipynb" notebook
you can use the datasets provided in cropped images just unzip the specific zip so the images and txt files are in the cropped images folder
