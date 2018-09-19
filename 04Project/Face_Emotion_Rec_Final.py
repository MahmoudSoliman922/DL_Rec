import face_recognition
import cv2
import requests
import boto3
import threading
import os
from urllib import urlencode
import urllib2
import urllib
import socket
from websocket import create_connection
import json
import numpy as np
from keras.preprocessing import image
import time
import random, string
import os
import sys

#------- Classes and functions
class people():
    face_found = False
    val = {
        'name': 'Unknown',
        'mood': 'Unknown',
        'imageName':'none',
        'eventName':'profile',
        'reactions': {
            'happy': '0',
            'sad': '0',
            'angry': '0',
            'calm': '0',
            'disgusted': '0',
            'confused': '0',
            'surprised': '0'
        }
    }


class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(
            self, group, target, name, args, kwargs, Verbose)
        self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)

    def join(self):
        threading.Thread.join(self)
        return self._return


def localEmotionRecognition(img):

    img = cv2.resize(img, (740, 560))
	# img = img[0:308,:]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        if w > 50:  # trick: ignore small faces
            # crop detected face
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]
            detected_face = cv2.cvtColor(
                detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
            detected_face = cv2.resize(
                detected_face, (48, 48))  # resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis=0)

            # pixels are in scale of [0, 255]. normalize all pixels in scale of
            # [0, 1]
            img_pixels /= 255
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            overlay = img.copy()
            opacity = 0.4
            cv2.rectangle(img, (x + w + 10, y - 25), (x + w + 150, y + 115),
                          (64, 64, 64), cv2.FILLED)
            cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

            # connect face and expressions
            cv2.line(img, (int((x + x + w) / 2), y + 15),
                     (x + w, y - 20), (255, 255, 255), 1)
            cv2.line(img, (x + w, y - 20),
                     (x + w + 10, y - 20), (255, 255, 255), 1)
            all_emotions_numbers = []
            for i in range(len(predictions[0])):
                emotion = "%s %s%s" % (emotions[i], round(
			        predictions[0][i] * 100, 2), '%')
                people.val['reactions'][emotions[i]] = round(predictions[0][i]*100, 2)
                all_emotions_numbers.append(round(predictions[0][i]*100, 2))
            people.val['mood'] = emotions[all_emotions_numbers.index(max(all_emotions_numbers))]


def randomword():
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(10))

def uploadToS3(passed_small_frame):
    temp_image_word=randomword()
    people.val['imageName'] = temp_image_word+'.jpg'
    people.val['name'] = temp_image_word
    print(people.val['imageName'])
    cv2.imwrite(filename='Faces2/'+people.val['imageName'], img=passed_small_frame)
    temp_image = face_recognition.load_image_file('Faces2/'+people.val['imageName'])
    temp_encoding = face_recognition.face_encodings(temp_image)[0]
    np.save('ImagesEncodings/'+temp_image_word+'.npy', temp_encoding)
    known_people_encodings.append(temp_encoding)
    known_people_name.append(temp_image_word)
    s3.Bucket(BUCKET).upload_file("/home/mahmoud/04Projects/04Project/Faces2/"+people.val['imageName'], people.val['imageName'],ExtraArgs={'ACL':'public-read'})

def emotionRecognition(passed_small_frame):
    cv2.imwrite(filename="temp.jpg", img=passed_small_frame)
    temp_image = "temp.jpg"
    with open(temp_image, 'rb') as image:
        response = client.detect_faces(Image={'Bytes': image.read()},
                                       Attributes=[
            'ALL'
        ]
        )
        image.close()
    maxEmotion = 0
    if response['FaceDetails'] != []:
        if response['FaceDetails'][0]['Emotions']:
            for item in response['FaceDetails'][0]['Emotions']:
                if item['Confidence'] > maxEmotion:
                    maxEmotion = item['Confidence']
                    people.val['mood'] = item['Type']
                    people.val['reactions'][item['Type'].lower()] = int(
                        item['Confidence'])
                else:
                    people.val['reactions'][item['Type'].lower()] = int(
                        item['Confidence'])
        else:
            print("No Emotions found!")
    else:
        people.val = {
            'name': 'Unknown',
            'mood': 'Unknown',
            'reactions': {
                    'happy': '0',
                    'sad': '0',
                    'angry': '0',
                    'calm': '0',
                    'disgusted': '0',
                    'confused': '0',
                    'surprised': '0'
            }
        }
        
        print("No faces found!")

#--------- End of Classes and functions

# -------------------- New package dependencies


# opencv initialization
face_cascade = cv2.CascadeClassifier(
    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

# -----------------------------
# face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open(
    "/home/mahmoud/04Projects/emotionRecTest/tensorflow-101/model/facial_expression_model_structure.json", "r").read())
model.load_weights(
    '/home/mahmoud/04Projects/emotionRecTest/tensorflow-101/model/facial_expression_model_weights.h5')  # load weights
# -----------------------------

emotions = ('angry', 'disgusted', 'confused',
            'happy', 'sad', 'surprised', 'calm')
# --------------------------- End of dependencies

# aws rekognition client declaration
client = boto3.client('rekognition')
# aws s3 bucket client declaration
s3Client = boto3.client('s3')
# bucket instance
s3 = boto3.resource('s3')
BUCKET = "face-rec-final"


#---------------- Code starts here
# Open a file
path = "ImagesEncodings/"
dirs = os.listdir(path)
known_people_name = []
known_people_encodings = []
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
temp_person = 'unknown'
temp_emotion = 'unknown'
# This would print all the files and directories
for file in dirs:
    fname, fext = file.split('.')
    known_people_name.append(fname)
    temp_encoding = np.load('ImagesEncodings/' + file)
    known_people_encodings.append(temp_encoding)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
ws = create_connection("ws://dev.getsooty.com:5555")
while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)


        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_people_encodings, face_encoding, tolerance=0.5)
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                localEmotionRecognition(frame)
                first_match_index = matches.index(True)
                name = known_people_name[first_match_index]
                people.val['name'] = name
                people.val['imageName'] = 'none'
                people.face_found = True
        if people.face_found == False and not face_encodings == []:
            localEmotionRecognition(frame)
            print('unknown person detected!')
            uploadImage = ThreadWithReturnValue(target=uploadToS3,args = (small_frame,))
            uploadImage.start()
        elif face_encodings == []:
            people.val = {
                'name': 'Unknown',
                'mood': 'Unknown',
                'imageName':'none',
                'eventName':'profile',
                'reactions': {
                    'happy': '0',
                    'sad': '0',
                    'angry': '0',
                    'calm': '0',
                    'disgusted': '0',
                    'confused': '0',
                    'surprised': '0'
                }
            }


        process_this_frame = not process_this_frame

        if temp_person != people.val['name'] or temp_emotion != people.val['mood']:
            ws.send(json.dumps(people.val))
            temp_person = people.val['name']
            temp_emotion = people.val['mood']
            print('New person or emotion detected!')
            print(people.val)
            people.face_found = False

# Release handle to the webcam
ws.close()
video_capture.release()
cv2.destroyAllWindows()
