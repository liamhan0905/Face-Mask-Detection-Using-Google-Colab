# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import tensorflow as tf
import numpy as np
import imutils
import time
import cv2
import os
import argparse
import sys
import time
from threading import Thread
import importlib.util



### get it working with just tflite
####https://www.youtube.com/watch?v=qJMwNHQNOVU (5:39) will tell how to convert it to tpu




import argparse
import sys
import time
from threading import Thread
import importlib.util


def detect_and_predict_mask(frame, face_interpreter, interpreter):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
<<<<<<< Updated upstream
	blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320),
		(104.0, 177.0, 123.0))

=======
	# blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320),
	# 	(104.0, 177.0, 123.0))
	frame = cv2.resize(frame, dsize=(320,320), interpolation=cv2.INTER_CUBIC)
>>>>>>> Stashed changes
	# pass the blob through the network and obtain the face detections
	# faceNet.setInput(blob)
	# detections = faceNet.forward()
	face_input_data = np.expand_dims(frame, axis=0)
	face_interpreter.set_tensor(face_input_details[0]['index'], face_input_data)
	face_interpreter.invoke()
	
	detections = face_interpreter.get_tensor(face_output_details[0]['index'])

	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	return locs
	# only make a predictions if at least one face was detected
	# if len(faces) > 0:
	# 	# for faster inference we'll make batch predictions on *all*
	# 	# faces at the same time rather than one-by-one predictions
	# 	# in the above `for` loop
	# 	# faces = np.array(faces, dtype="float32")

	# 	faces_img = np.array(faces, dtype="float32")
	# 	input_data = faces_img
	# 	# input_data = np.expand_dims(faces_img, axis=0)
		

	# 	interpreter.set_tensor(input_details[0]['index'], input_data)
	# 	interpreter.invoke()
		
	# 	preds = interpreter.get_tensor(output_details[0]['index'])

	# 	# preds = maskNet.predict(faces, batch_size=32)

	# # return a 2-tuple of the face locations and their corresponding
	# # locations
	# return (locs, preds)

# load our serialized face detector model from disk
# prototxtPath = "./face_detector/deploy.prototxt"
# weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
use_TPU = True
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

face_path = "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"

use_TPU = True
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# face_interpreter = tf.lite.Interpreter(face_path)
face_interpreter = tf.lite.Interpreter(model_path=face_path,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
face_interpreter.allocate_tensors()

face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()



maskNet_path = "mask_detector.tflite"

interpreter = tf.lite.Interpreter(maskNet_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()






# initialize the video stream
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)


cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = cap.read()

	# frame = imutils.resize(frame, width=400)
	# print(frame)
	
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	# (locs, preds) = detect_and_predict_mask(frame, face_interpreter, interpreter)
	locs = detect_and_predict_mask(frame, face_interpreter, interpreter)
	# loop over the detected face locations and their corresponding
	# locations
	# for (box, pred) in zip(locs, preds):
	for box in locs:
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box

		# mask = pred[0]
		# withoutMask = pred[1]
		# # (mask, withoutMask) = pred

		# # determine the class label and color we'll use to draw
		# # the bounding box and text
		# label = "Mask" if mask > withoutMask else "No Mask"
		# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# # include the probability in the label
		# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		# cv2.putText(frame, label, (startX, startY - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.putText(frame, "test", (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
# vs.stop()
