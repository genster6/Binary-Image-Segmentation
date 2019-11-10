
import traceback
import random
import numpy as np
import skimage.measure
import cv2
import copy
from PIL import Image
import time #
import pandas as pd#
from skimage import color#
import os#
import sys#

from . import AlgorithmParams
from . import AlgorithmSpace
from . import AlgorithmHelper
from .AlgorithmHelper import AlgoHelp


class GeneticHelp(object):

	#Executes a crossover between two numpy arrays of the same length
	def twoPointCopy(np1, np2):
		assert(len(np1) == len(np2))
		size = len(np1)
		point1 = random.randint(1, size)
		point2 = random.randint(1, size-1)
		if (point2 >= point1):
			point2 +=1
		else: #Swap the two points
			point1, point2 = point2, point1
		np1[point1:point2], np2[point1:point2] = np2[point1:point2].copy(), np1[point1:point2].copy()
		return np1, np2

	'''Executes a crossover between two arrays (np1 and np2) picking a 
	random amount of indexes to change between the two.
	'''
	def skimageCrossRandom(np1, np2):
		#TODO: Only change values associated with algorithm
		assert(len(np1) == len(np2))
		#The number of places that we'll cross
		crosses = random.randrange(len(np1))
		#We pick that many crossing points
		indexes = random.sample(range(0, len(np1)), crosses)
		#And at those crossing points, we switch the parameters
		
		for i in indexes:
			np1[i], np2[i] = np2[i], np1[i]

		return np1, np2

	''' Changes a few of the parameters of the weighting a random
		number against the flipProb.
		Variables:
		copyChild is the individual to mutate
		posVals is a list of lists where each list are the possible
			values for that particular parameter
		flipProb is how likely, it is that we will mutate each value.
			It is computed seperately for each value. 
	'''
	def mutate(copyChild, posVals, flipProb = 0.5):

		#Just because we chose to mutate a value doesn't mean we mutate
		#Every aspect of the value	
		child = copy.deepcopy(copyChild)

		#Not every algorithm is associated with every value
		#Let's first see if we change the algorithm
		randVal = random.random()
		if randVal < flipProb:
			#Let's mutate
			child[0] = random.choice(posVals[0])
		#Now let's get the indexes (parameters) related to that value
		switcher = AlgoHelp().algoIndexes()
		indexes = switcher.get(child[0])

		for index in indexes:
			randVal = random.random()
			if randVal < flipProb:
				#Then we mutate said value
				if index == 22:
					#Do some special
					X = random.choice(posVals[22])
					Y = random.choice(posVals[23])
					Z = random.choice(posVals[24])
					child[index] = (X, Y, Z)
					continue

				child[index] = random.choice(posVals[index])
		return child		

	'''
	function to calculate number of sets in our test image 
	that map to more than one set in our truth image, and how many
	pixels are in those sets. Used in fitness function below.
	INPUTS: truth image, infer image
	RETURNS: number of repeated sets, number of pixels in repeated sets
	'''
	def set_fitness_func(a_test, b_test, include_L = False):
		a_test_int = a_test.ravel().astype(int) # turn float array into int array
		b_test_int = b_test.ravel().astype(int) # turn float array into in array

		# create char array to separate two images
		filler = np.chararray((len(a_test_int))) 
		filler[:] = ':'

		# match arrays so we can easily compare
		matched = np.core.defchararray.add(a_test_int.astype(str), filler.astype(str))
		matched = np.core.defchararray.add(matched, b_test_int.astype(str))

		# collect unique set pairings
		unique_sets = np.unique(matched)

		# count number of pixels for each set pairing
		set_counts = {}
		for i in unique_sets:
			set_counts[i] = sum(np.core.defchararray.count(matched, i))
		
		# print statements for debugging
	#     print('UNIQUE: ', unique_sets) # see set pairings
	#     print('SET_COUNTS: ', set_counts) # see counts
		
		## counts every repeated set. EX: if we have (A, A, B, B, B, C) we get 5 repeated. 
		sets = set() # init container that will hold all sets in infer. image
		repeats = [] # init container that will hold all repeated sets
		b_set_counts = {} # init container that will hold pixel counts for each repeated set
		for i in unique_sets:
			current_set = i[i.find(':')+1:] # get inf. set from each pairing
			if current_set in sets: # if repeat set
				repeats.append(current_set) # add set to repeats list
				b_set_counts[current_set].append(set_counts[i]) # add pixel count to set in dict.
			elif current_set not in sets: # if new set
				b_set_counts[current_set] = [set_counts[i]] # init. key and add pixel count
				sets.add(current_set) # add set to sets container

		# get number of repeated sets
		num_repeats = len(np.unique(repeats)) + len(repeats) 
		# num_repeats = len(sets)## get all sets in infer image

		# count number of pixels in all repeated sets. Assumes pairing with max. num 
		# of pixels is not error
		repeat_count = 0
		used_sets = set()
		for i in b_set_counts.keys():
			repeat_count += sum(b_set_counts[i]) - max(b_set_counts[i])
			for j in unique_sets:
				if j[j.find(':')+1:] == i and set_counts[j] == max(b_set_counts[i]):
					used_sets.add(j[:j.find(':')])
			
		if include_L == True:
			return num_repeats, repeat_count, used_sets
		else:
			return num_repeats, repeat_count

	'''Takes in two ImageData obects and compares them according to
	skimage's Structual Similarity Index and the mean squared error
	Variables:
	img1 is an image array segmented by the algorithm. 
	img2 is the validation image
	imgDim is the number of dimensions of the image.
	'''
	def __FitnessFunction(img1, img2, imgDim):	
		# assert(len(img1.shape) == len(img2.shape) == imgDim)

		# #The channel deterimines if this is a RGB or grayscale image
		# channel = False
		# if imgDim > 2: channel = True
		# #print(img1.dtype, img2.dtype)
		# img1 = np.uint8(img1)
		# #print(img1.dtype, img2.dtype)
		# assert(img1.dtype == img2.dtype)
		# #TODO: Change to MSE
		# #Comparing the Structual Similarity Index (SSIM) of two images
		# ssim = skimage.measure.compare_ssim(img1, img2, win_size=3, 
		# 	multichannel=channel, gaussian_weights=True)
		# #Comparing the Mean Squared Error of the two image
		# #print("About to compare")
		# #print(img1.shape, img2.shape, imgDim)
		# #mse = skimage.measure.compare_mse(img1, img2)
		# #Deleting the references to the objects and freeing memory
		# del img1
		# del img2
		# #print("eror above?")
		# return [abs(ssim),]

		# makes sure images are in grayscale
		if len(img1.shape) > 2: 
			img1 = color.rgb2gray(img1)
		if len(img2.shape) > 2: ## comment out
			img2 = color.rgb2gray(img2) ## comment out
		# img2 = img2[:,:,0]#color.rgb2gray(true_im) # convert to grayscale
		# img2[img2[:,:] != 0] = 1
		# makes sure images can be read as segmentation labels (i.e. integers)
		img1 = pd.factorize(img1.ravel())[0].reshape(img1.shape)
		img2 = pd.factorize(img2.ravel())[0].reshape(img2.shape)

		num_repeats, repeat_count, used_sets = GeneticHelp.set_fitness_func(img2, img1, True)
		m = len(np.unique(img1))
		n = len(np.unique(img2))
		L = len(used_sets)
		error = (repeat_count + 2)**np.log(abs(m - n)) / (L >= n)
		# error = (repeat_count + 2)**(abs(m - n)+1)
		if error <= 0 or error == np.inf or error == np.nan:
			error = sys.maxsize
			# print(error)
		return [error,]

	'''Runs an imaging algorithm given the parameters from the 
		population
	Variables:
	copyImg is an ImageData object of the image
	valImg is an ImageData object of the validation image
	individual is the parameter that we chose
	'''
	def runAlgo(copyImg, groundImg, individual):
		init = time.time()
		img = copy.deepcopy(copyImg)
		#Making an AlorithmParams object
		params = AlgorithmParams.AlgorithmParams(img, individual)


		Algo = AlgorithmSpace.AlgorithmSpace(params)

		#Python's version of a switch-case
		#Listing all the algorithms. For fun?
		AllAlgos = [
			'RW',#: Algo.runRandomWalker,
			'FB',#: Algo.runFelzenszwalb,
			'SC',#: Algo.runSlic,
			'QS',#: Algo.runQuickShift,
			'WS',#: Algo.runWaterShed,
			'CV',#: Algo.runChanVese,
			'MCV',#: Algo.runMorphChanVese,
			'AC',#: Algo.runMorphGeodesicActiveContour,
			'FD',#: Algo.runFlood,
			'FF',#: Algo.runFloodFill
		]
		#Some algorithms return masks as opposed to the full images
		#The functions in Masks and BoolArrs will need to pass through
		#More functions before they are ready for the fitness function
		switcher = AlgoHelp().channelAlgos(img)


		# If the algorithm is not right for the image, return large number
		if (params.getAlgo() not in switcher): return [sys.maxsize,]#[100,]
		
		#Running the algorithm and parameters on the image
		try:
		    runAlg = AlgorithmSpace.AlgorithmSpace(params)
		    img = runAlg.runAlgo() # takes a long time ALGORITHMSPACE FUNCTION
		    		#The algorithms in Masks and BoolArrs need to be applied to the
		    #	img
		    
		    #Running the fitness function
		    evaluate = GeneticHelp.__FitnessFunction(np.array(img), groundImg.getImage(), len(np.array(img).shape))	
		except KeyboardInterrupt as e:
                    raise e
		except:
		    e = sys.exc_info()[0]
		    print(f"ERROR: {e}") 
		    traceback.print_exc() 
		    evaluate = [ 9999999999999999, ] 

		return (evaluate)
