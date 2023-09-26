# import the necessary packages
import numpy as np
import imutils
import cv2
 
class SingleMotionDetector:
	def __init__(self, accumWeight=0.5):
		# store the accumulated weight factor
		self.accumWeight = accumWeight
 
		# initialize the background model
		self.bg = None