# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from flask import Response
from flask import Flask
from flask import render_template
import threading
import datetime
import numpy as np
import argparse
import imutils
import pickle
import time
import requests 
import cv2
import os
import pyttsx3
import freenect
import frame_convert2

import socket
import sys
from enum import Enum 
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
ledColor = enum("off", "green", "red", "orange", "tGreen", "tGreen2", "redOrange")

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the port
server_address = ('0.0.0.0', 10000)
print(sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)
# Listen for incoming connections
sock.listen(1)

threshold = 42 # 45
current_depth = 805 #920
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
 
# initialize a flask object
app = Flask(__name__)
 



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")

ap.add_argument("-i", "--ip", required=True,default="0.0.0.0",
	help="ip")
ap.add_argument("-p", "--port", required=True,
	help="port")
ap.add_argument("-s", "--camera", required=True,
	help="camera source")

ap.add_argument("-o", "--output", required=True,
	help="path to output directory")

args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read(),encoding="latin-1")
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
kinect = False
if(args["camera"] == 'kinect'):
	kinect = True
if(not kinect):
	vs = VideoStream(src=0).start()
	time.sleep(2.0)


# loop over frames from the video file stream
count = 0
pause = 0

connection = None
orig = None
def recognize_face():
	global outputFrame,lock,connection,orig
	# grab the frame from the threaded video stream
	last = ""
	count = 0
	voicetime = 0
	failDetections = 0
	lon = False
	while (1):
		depth, timestamp = freenect.sync_get_depth()
		depth = 255 * np.logical_and(depth >= current_depth - threshold,
	                                 depth <= current_depth + threshold)
		depth = depth.astype(np.uint8)
		bits = 0
		for fila in depth:
			for b in fila[100:]:
				if(b):
					bits+=1
		if(bits < 25000):
			pass
			print("nop " + str(bits) )
			failDetections = 0
			freenect.sync_set_led(ledColor.green,0)
			continue
		else:
			print("yes")
			if(count == 0):
				freenect.sync_set_led(ledColor.red,0)
				if(voicetime == 0):
					engine = pyttsx3.init()
					engine.say('please wait while the system scan you')
					engine.runAndWait()
					voicetime = 50
				else:
					voicetime -= 1
			

		if(kinect):
			frame = frame_convert2.video_cv(freenect.sync_get_video()[0])
		else:
			frame = vs.read()
		orig = frame.copy()
		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > args["confidence"]:
				failDetections = 0
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				print("cara")
				if(proba > 0.8 and name != 'unknown'):
					#if(connection):
					#	data = (name + ' a entrado').encode('ascii')
					#	connection.send(data)
					#requests.get("http://192.168.0.110/on?code=abrir", "") 
					if(last == name):
						count += 1
					else:
						count = 0
					last = name
					if(count > 5):
						count = 0
						print(name)
						freenect.sync_set_led(ledColor.green,0)
						engine = pyttsx3.init()
						engine.say('access denied to ' + name)
						engine.runAndWait()
						freenect.sync_set_led(ledColor.tGreen,0)
						#os.system("")
						if(name == "david"):
							os.system("curl 192.168.0.140/alarma")
							time.sleep(5)
							os.system("curl 192.168.0.140/apagar")
							
							pass
							#os.system("ssh -p 21 192.168.0.121 aset on")
							#os.system("ssh -p 21 192.168.0.121 ledset power")
							#os.system("ssh -p 21 192.168.0.121 ledset fade3")
							
						else:
							time.sleep(5)
				
				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			else:
				if(failDetections > 200 and not lon):
					failDetections = 0
					lon = True
					#os.system("ssh -p 21 ley@192.168.0.121 lon")
				else:
					failDetections += 1
		# update the FPS counter

		# show the output frame
		with lock:
			outputFrame = frame.copy()
		#cv2.imshow("Frame", frame)

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock, orig
	# loop over frames from the output stream
	total = 0
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
 
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			cv2.imshow("Frame", outputFrame)
			key = cv2.waitKey(1) & 0xFF
 
			if key == ord("k"):
				p = os.path.sep.join([args["output"], "{}.png".format(
					str(total).zfill(5))])
				cv2.imwrite(p, orig)
				total +=1
 
			# ensure the frame was successfully encoded
			if not flag:
				continue
 
		# yield the output frame in the byte format
		
		yield(b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def tcpserverloop():
	global connection
	while(1):
	    print (sys.stderr, 'waiting for a connection')
	    connection, client_address = sock.accept()
	    print('conected to ',client_address)


# start a thread that will perform motion detection
t = threading.Thread(target=recognize_face, args=())
t.daemon = True
t.start()
#t2 = threading.Thread(target=tcpserverloop,args=())
#t2.daemon = True
#t2.start()
app.run(host=args["ip"], port=args["port"], debug=True,
	threaded=True, use_reloader=False)

#cv2.destroyAllWindows()