import cv2
import os
from PIL import Image
import numpy as np
import sys
import time
#from . import ImageData

class FileClass(object):
	#Writes an image to a file
	#Variables
	#img is an ImageData object, 
	#fileName is the name of the file

	#https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
	#Image.fromarray doesn't like boolean arrays. we first make a grayscale
	#Returns a black/white Image based on a boolean numpy array. 
	def imgFromBytes(data):
		size = data.shape[::1]
		databytes = np.packbits(data, axis=1)
		return Image.frombytes(mode='1', size=size, data=databytes)

	#Writes the text representation of an image to a file
	def writeImage(img, fileName):
		cv2.imwrite(fileName, img.getImage())

	#Converts an image mask to a multichannel image
	def convertMask(imgArr):
		multiImg = np.ndarray(imgArr.shape[0] * imgArr.shape[1] * 3)
		multiImg.shape = (imgArr.shape[0], imgArr.shape[1], 3)
		
		#0 is black, 134 is red
		for le in range(0, len(imgArr)):
			for wi in range(0, len(imgArr[0])):
				#Filling in the black
				if imgArr[le][wi] == 0 or imgArr[le][wi] == False:
					multiImg[le][wi] = [0,0,0]
				if imgArr[le][wi] == 1 or imgArr[le][wi] == True:
					multiImg[le][wi] = [0,0,134]
				else:
					multiImg[le][wi] = [0,0,0]
		return multiImg

	#img is an ImageData object
	#txtName is the name of the file
	def writeData(img, txtName):
		file = open(txtName, 'w+')
		for line in img.getImage():
			for number in line:
				file.write(str(number) + " ")
			file.write('\n')
		file.write(str(img.getImage()))
		file.close()

	#Checks if a directory exists where 'path' is said directory
	def check_dir(path):
		directory = os.path.dirname(path)
		if not os.path.exists(path):
			return False
		return True

	#Checks if a directory exists. If it doesn't creates
	#path is said directory
	def check_and_create(path):
		if (check_dir(path) == False):
			os.makedirs(path)

	#Given a path, finds the type of image it is (e.g. .png, .jpeg)
	def findImageType(path):
		imgType = ""
		isImage = False
		for i in range(0, len(path)):
			if isImage == True:
				imgType += path[i]
			elif path[i] == '.':
				imgType = True
		return imgType


