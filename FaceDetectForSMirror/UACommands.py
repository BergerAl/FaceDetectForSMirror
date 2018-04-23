#import picamera
#import cv2
#import time
from numpy import reshape
import os
import json

class UserAnalysis():

    def loadCascadeAndModel():
        #Face recognition cascade
        self.faceCascade = cv2.CascadeClassifier('models/lbpcascade_frontalface.xml')

        #load model and weights
        json_file = open('models/basic_cnn_30_epochs_data.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        self.model.load_weights("models/basic_cnn_30_epochs_data.h5")
        self.model_done = True

        print ("Everything was loaded as intended")

    def takePictureAndScale():
        image_name = 'CameraPics/' + date + '.jpg'
        camera.capture(image_name)
        #read to array and resize
        image = cv2.imread(image_name)
        r = 600.0/image.shape[1]
        dim = (600, int(image.shape[0]*r))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        #add grayscale
        grayScaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayScaleImage

    def extractFacesFromImage(imageAsArray):
        loadCascadeAndModel()
        # Detect faces in the image
        faces = self.faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                flags = cv2.CASCADE_SCALE_IMAGE
                )
        if faces == ():
            print("No faces detected")
            time.sleep(3)
        else:
            #load list of persons from json

            #Prepare lsit of persons in image
            person_list = []
            for (x, y, w, h) in faces:

                boundary_factor = 0.1
                # Extracting face from image
                recog = image[y-int(boundary_factor*h):y+int(h*(1+boundary_factor)), x-int(boundary_factor*w):x+int(w*(1+boundary_factor))]
                ### Analysing in CNN ###
                recog = cv2.resize(recog, (150,150))
                image_as_array = img_to_array(recog)
                image_as_array = image_as_array.reshape((1,) + image_as_array.shape)
                #need a bigger array or something
                prediction = self.model.predict(image_as_array)
                print(prediction)
                person_list.append(self.name_assignment(prediction))


    def main(self, GpioChannel):
        print("let's go")

class addingNewPerson():

    def __init__(self):
        #ISSUE: read path fo Configuration.json
        self.jsonConfigPath = "C:/Users/Aleks/Desktop/FaceDetectForSMirror/FaceDetectForSMirror/user/Configutation.json"
        self.pictureAmount = 100

    def gettingPersonsName(self):
        self.personsName = input("Enter Persons Name: ")
        #Unsolved issue: double Names?
        pictureDir = "CameraPics/ModelTraining/" + personsName
        if not os.path.exists(pictureDir):
            os.makedirs(pictureDir)

    def trainModel():
        a=1

    def addConfigDetails(self):
            self.gettingPersonsName()
            wholeDataFile = json.loads(open(self.jsonConfigPath).read())
            #default entry
            entry = {self.personsName : [{"App1":{"config": [{"position" : 28.2},{"high" : 22.3},{"width" : 11.2}]}},{"App2": 0}]}
            wholeDataFile['user'].append(entry)
            #writing to file
            with open(self.jsonConfigPath, 'w') as outfile:
                json.dump(wholeDataFile, outfile, indent=2)

    def takePictiresForModelTraining(self):
        for i in range(0,self.pictureAmount-1):
            image_name = "CameraPics/ModelTraining/" + personsName +"/" + i + '.jpg'
            camera.capture(image_name)
