#Strongly coupled with AlgorithmParams and AlgorithmHelper
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

#TODO: Add color segmetation.  

import numpy as np
import pandas as pd
import copy
import skimage
from skimage import segmentation
from itertools import combinations
import cv2
from . import FileClass
from .FileClass import FileClass
from .AlgorithmHelper import AlgoHelp
from PIL import Image
import time #
#from . import ImageData
#from . import AlgorithmParams

'''
This class will run through all the algorithms in skimage.segmentation
and change the parameters
'''

class AlgorithmSpace(object):
	def __init__(self, parameters):
		#parameters is a AlgorithmParams object
		assert(type(parameters).__name__ == "AlgorithmParams")
		self.params = parameters
		#Whether this is a multichannel array or grayscale
		self.channel = False
		if (self.params.getImage().getDim() > 2):
			#This is at least a 3D array, so multichannel
			self.channel = True
		self.newVal = 134

	

	#Algorithms
	'''
	#Random walker algorithm: 
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html
	Segments an image from set markers. Works
	on grayscale and multichannel images,

	#Returns a labeled image (ndarray)
	#Variables:
	data -> Image (ndarray)
	labels -> Same shape as data, ndarray
	beta -> float, penalization coefficient High beta = more difficult
	diffusion. as different coefficients
	may be better for different images
	mode -> string. mode is normally bf, but should use cg_mg for 
		images smaller than 512X512
	tol is the tolerance to achieve when solving a linear system
	copy should be True. Whether or not to overwrite label array
	multichannel -- False is gray image, True if multichannel image
	return_full_prob -> should be false
	spacing -> spacing in between pixels, Going to leave this at 1 for 
		now

	'''
	#Code for algorithms == RW
	#Not using RandomWalker because labels complicates the searchspace
		
	def __runRandomWalker(self, mask):
		#Let's deterime what mode to use
		mode = ""
		if len(self.params.getImage().getImage()) < 512 :
			mode = "cg_mg"
		else:
			mode = "bf"
		
		#If data is 2D, then this is a grayscale, so multichannel is 
		#false
		
		output = skimage.segmentation.random_walker(
			self.params.getImage().getImage(), labels=mask,
			beta=self.params.getBeta(), 
			tol=self.params.getTolerance(), copy=True, 
			multichannel=True, return_full_prob=False) 
		return output

	'''
	#felzenszwalb
	#ONLY WORKS FOR RGB
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	The felzenszwalb algorithms computes a graph based on the 
		segmentation
	Produces an oversegmentation of the multichannel using min-span
		tree.
	Returns an integer mask indicating the segment labels
	#Variables
	data : Image (ndarray)
	scale: float, higher meanse larger clusters
	sigma: float, std. dev of Gaussian kernel for preprocessing
	min_size: int, minimum component size. For postprocessing
	mulitchannel: bool, Whether the image is 2D or 3D. 2D images
		are not supported at all


	'''
	#Abbreviaiton for algorithm == FB

	def __runFelzenszwalb(self):
		output = skimage.segmentation.felzenszwalb(
			self.params.getImage().getImage(), self.params.getScale(),
			self.params.getSigma(), self.params.getMinSize(),
			multichannel=self.channel)
		
		return output

	#Needs testing
	'''
	#slic
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	segments k-means clustering in Color space (x, y, z)
	#Returns a 2D or 3D array of labels

	#Variables
	image -- ndarray, input image
	n_segments -- int,  number of labels in segmented output image 
		(approx). Should find a way to compute n_segments
	compactness -- float, Balances color proximity and space proximity.
		Higher values mean more weight to space proximity (superpixels
		become more square/cubic) #Recommended log scale values (0.01, 
		0.1, 1, 10, 100, etc)
	max_iter -- int, max number of iterations of k-means
	sigma -- float or (3,) shape array of floats,  width of Guassian
		smoothing kernel. For pre-processing for each dimesion of the
		image. Zero means no smoothing
	spacing -- (3,) shape float array : voxel spacing along each image
		dimension. Defalt is uniform spacing
	multichannel -- bool,  multichannel (True) vs grayscale (False)
	#Needs testing to find correct values
	
	#Abbreviation for algorithm == SC
	'''
	def __runSlic(self):
		output = skimage.segmentation.slic(
			self.params.getImage().getImage(),
			n_segments=self.params.getSegments(), compactness=
			self.params.getCompact(), max_iter=self.params.getIters(),
			sigma=self.params.getSigma(), multichannel=self.channel)

		return output

	'''
	#quickshift
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	Segments images with quickshift clustering in Color (x,y) space
	#Returns ndarray segmentation mask of the labels
	#Variables
	image -- ndarray, input image
	ratio -- float, balances color-space proximity & image-space
		proximity. Higher vals give more weight to color-space
	kernel_size: float, Width of Guassian kernel using smoothing.
		Higher means fewer clusters
	max_dist -- float: Cut-off point for data distances. Higher 
		means fewer clusters
	return_tree -- bool: Whether to return the full segmentation
		hierachy tree and distances. Set as False
	sigma -- float: Width of Guassian smoothing as preprocessing.
		Zero means no smoothing
	convert2lab -- bool: leave alone
	random_seed -- int, Random seed used for breacking ties. 
	'''
	#Abbreviation for algorithm == QS

	def __runQuickShift(self):
		output = skimage.segmentation.quickshift(
			self.params.getImage().getImage(), 
			ratio=self.params.getRatio(), 
			kernel_size=self.params.getKernel(), max_dist=
			self.params.getMaxDist(), sigma=self.params.getSigma(),
			random_seed=self.params.getSeed())
		
		return output


	'''
	#Watershed
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
	Uses user-markers. treats markers as basins and 'floods' them.
	Especially good if overlapping objects. 
	#Returns a labeled image ndarray
	#Variables
	image -> ndarray, input array
	markers -> int, or int ndarray same shape as image: markers 
		indicating 'basins'
	connectivity -> ndarray, indicates neighbors for connection
	offset -> array, same shape as image: offset of the connectivity
	mask -> ndarray of bools (or 0s and 1s): 
	compactness -> float, compactness of the basins Higher values make 
		more regularly-shaped basin
	'''
	#Not using connectivity, markers, or offset params as arrays would
	#expand the search space too much.
	#abbreviation for algorithm = WS

	def __runWaterShed(self):
		output = skimage.segmentation.watershed(
			self.params.getImage().getImage(),markers=None,
			compactness=self.params.getCompact())
		return output
	
	'''
	#chan_vese
	#ONLY GRAYSCALE
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_chan_vese.html
	Segments objects without clear boundaries
	#Returns: segmentation array of algorithm. Optional: When the 
		algorithm converges
	#Variables
	image -> ndarray grayscale image to be segmented
	mu -> float, 'edge length' weight parameter. Higher mu vals make a 
		'round edge'
		closer to zero will detect smaller objects
	lambda1 -> float 'diff from average' weight param to determine if 
		output region is True. If lower than lambda1, the region has a 
		larger range of values than the other
	lambda2 -> float 'diff from average' weight param to determine if 
		output region is False. If lower than lambda1, the region will 
		have a larger range of values
	Note: Typical values for mu are from 0-1. 
	Note: Typical values for lambda1 & lambda2 are 1. If the background 
		is 'very' different from the segmented values, in terms of
		distribution, then the lambdas should be different from 
		eachother
	tol: positive float, typically (0-1), very low level set variation 
		tolerance between iterations.
	max_iter: uint,  max number of iterations before algorithms stops
	dt: float, Multiplication factor applied at the calculations step
	init_level_set: str/ndarray, defines starting level set used by
		algorithm. Accepted values are:
		'checkerboard': fast convergence, hard to find implicit edges
		'disk': Somewhat slower convergence, more likely to find
			implicit edges
		'small disk': Slowest convergence, more likely to find implicit
			edges
		can also be ndarray same shape as image
	extended_output: bool, If true, adds more returns 
		(Final level set & energies)
	'''
	#Abbreviation for Algorithm = CV
	
	def __runChanVese(self):
		output = skimage.segmentation.chan_vese(
			self.params.getImage().getImage(), mu=self.params.getMu,
				lambda1=self.params.getLambdaOne(), lambda2=
				self.params.getLambdaTwo(), 
				tol=self.params.getTolerance(),
				max_iter=self.params.getIters(), dt=self.params.getDT()
				, init_level_set=self.params.getInitLvlSetChan())

		return output
	'''

	#morphological_chan_vese
	#ONLY WORKS ON GRAYSCALE
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese
	Active contours without edges. Can be used to segment images/
		volumes without good borders. Required that the inside of the
		object looks different than outside (color, shade, darker).
	#Returns Final segmention
	#Variables:
	image -> ndarray of grayscale image
	iterations -> uint, number of iterations to run
	init_level_set: str, or array same shape as image. Accepted string
		values are:
		'checkerboard': Uses checkerboard_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set
		returns a binary level set of a checkerboard
		'circle': Uses circle_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.circle_level_set
		Creates a binary level set of a circle, given a radius and a
		center

	smoothing: uint, number of times the smoothing operator is applied
		per iteration. Usually around 1-4. Larger values make stuf 
		smoother
	lambda1: Weight param for outer region. If larger than lambda2, 
		outer region will give larger range of values than inner value
	lambda2: Weight param for inner region. If larger thant lambda1, 
		inner region will have a larger range of values than outer region
	'''
	#Abbreviation for algorithm = MCV

	def __runMorphChanVese(self):
		'''print("L2: ", self.params.getLambdaTwo())
		print("L1: ", self.params.getLambdaOne())
		print("Smoothing: ", self.params.getSmoothing())
		print("Init: ", self.params.getInitLvlSetMorph())
		print("iters: ", self.params.getIters())'''
		output = skimage.segmentation.morphological_chan_vese(
			self.params.getImage().getImage(), 
			iterations=self.params.getIters(),
			init_level_set=	self.params.getInitLvlSetMorph(), 
			smoothing=self.params.getSmoothing(), 
			lambda1=self.params.getLambdaOne(), 
			lambda2=self.params.getLambdaTwo())
		return output

	'''
	#morphological_geodesic_active_contour
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour
	Uses an image from inverse_gaussian_gradient in order to segment
		object with visible, but noisy/broken borders
	#inverse_gaussian_gradient
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.inverse_gaussian_gradient
	Compute the magnitude of the gradients in an image. returns a
		preprocessed image suitable for above function
	#Returns ndarray of segmented image
	#Variables
	gimage: array, preprocessed image to be segmented
	iterations: uint, number of iterations to run
	init_level_set: str, array same shape as gimage. If string, possible
		values are:
		'checkerboard': Uses checkerboard_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set
		returns a binary level set of a checkerboard
		'circle': Uses circle_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.circle_level_set
		Creates a binary level set of a circle, given a radius and a 
		center
	smoothing: uint, number of times the smoothing operator is applied 
		per iteration. Usually 1-4, larger values have smoother 
		segmentation
	threshold: Areas of image with a smaller value than the threshold
		are borders
	balloon: float, guides contour of low-information parts of image, 	
	'''
	#Abbrevieation for algorithm = AC

	def __runMorphGeodesicActiveContour(self):
		#We run the inverse_gaussian_gradient to get the image to use
		gimage = skimage.segmentation.inverse_gaussian_gradient(
			self.params.getImage().getImage(), self.params.getAlpha(), 
			self.params.getSigma())
		zeros = 0
		'''print("gimage")
		print("alpha: ", self.params.getAlpha())
		print("sigma: ", self.params.getSigma())
		print("image dims ", self.params.getImage().getDim())
		print("Image shape: ", self.params.getImage().getImage().shape)
		print("gimage shape: ", gimage.shape)
		'''
		output = skimage.segmentation.morphological_geodesic_active_contour(
			gimage, self.params.getIters(), 
			self.params.getInitLvlSetMorph(),
			smoothing= self.params.getSmoothing(), 
			threshold=self.params.getThresh(), balloon= 
			self.params.getBalloon())
		#print("Done")
		#print("output shape: ", output.shape)
		return output
	'''
	#flood
	#DOES NOT SUPPORT MULTICHANNEL IMAGES
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
	Uses a seed point and to fill all connected points within/equal to
		a tolerance around the seed point
	#Returns a boolean array with 'flooded' areas being true
	#Variables
	image: ndarray, input image
	seed_point: tuple/int, x,y,z referring to starting point for flood 
		fill
	selem: ndarray of 1's and 0's, Used to determine neighborhood of
		each pixel
	connectivity: int, Used to find neighborhood of each pixel. Can use 
		this or selem.
	tolerance: float or int, If none, adjacent values must be equal to 
		seed_point. Otherwise, how likely adjacent values are flooded.
	'''
	#Abbreviation for algorithm = FD

	def __runFlood(self):
		output = skimage.segmentation.flood(
			self.params.getImage().getImage(),
			self.params.getSeedPoint(), connectivity=
			self.params.getConnect(), tolerance=
			self.params.getTolerance())
		return output

	'''
	#flood_fill
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
	Like a paint-bucket tool in paint. Like flood, but changes the 
		color equal to new_type
	#Returns A filled array of same shape as the image
	#Variables
	image: ndarray, input image
	seed_point: tuple or int, starting point for filling (x,y,z)
	new_value: new value to set the fill to (e.g. color). Must agree
		with image type
	selem: ndarray, Used to find neighborhood of filling
	connectivity: Also used to find neighborhood of filling if selem is
		None
	tolerance: float or int, If none, adjacent values must be equal to 
		seed_point. Otherwise, how likely adjacent values are flooded.
	inplace: bool, If true, the flood filling is applied to the image,
		if False, the image is not modified. Default False, don't 
		change
	'''
	#Abbreviation for algorithm == FF

	def __runFloodFill(self):

		output = skimage.segmentation.flood_fill(
			self.params.getImage().getImage(), 
			self.params.getSeedPoint(), self.newVal, connectivity=
			self.params.getConnect(), tolerance=
			self.params.getTolerance())
		try:
			#I'm not sure if this will work on grayscale
			image = Image.fromarray(output.astype('uint8'), '1')
		except ValueError:
			image = Image.fromarray(output.astype('uint8'), 'RGB')

		width = image.width
		height = image.width


		#Converting the background to black
		for x in range(0, width):
			for y in range(0, height):
				#First check for grayscale
				pixel = image.getpixel((x,y))
				if pixel[0] == self.newVal:
					image.putpixel((x,y), self.newVal)
					continue
				else:
					image.putpixel((x,y), 0)
					#print(image.getpixel((x,y)))
#					print("black: ", image.getpixel((x,y)))

		#image.convert(mode='L')
		pic = np.array(image)
		return pic

	'''
	#mark_boundaries
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries	
	Some functions return masks of data, but not the actual images. 
	This will apply that mask to the data
	#Returns a marked array which corresponds to the labels
	#Variables
	image: 3D array, grayscale or RGB image
	label_img: 2D array, regions are different integer values
	color: RGB color sequence (length-3), optional
	outline_color: RGB color sequence (length-3), optional
	mode: string in {thick, inner, outer, subpixel} to define 
	boundaries, optional
	background_label: int, which label to consider background, optional
		Only useful in inner and outer mode
	'''
	#Abbreviations for algorithm == MB
	#Probably want to pass in color
	'''This is not an image segmentation algorithm on its own. 
	It requires
		another algorithm to create a mask first
	'''
	#Uses Felzenszwalb, slic,

	def __runMarkBoundaries(self, mask):
		output = skimage.segmentation.mark_boundaries(
			self.params.getImage().getImage(), mask, mode='inner',
			background_label=134)
		return output

	def __runClearBorder(self, mask):
		output = skimage.segmentation.clear_border(
			labels=mask, bgval=134)
		return output


	### ADD NEW ALGORITHMS HERE	

	#Runs the algorithm specified in params
	def runAlgo(self):
		init = time.time()
		switcher = {
			'RW': self.__runRandomWalker,
			'FB': self.__runFelzenszwalb,
			'SC': self.__runSlic,
			'QS': self.__runQuickShift, # takes a long time
			'WS': self.__runWaterShed,
			'CV': self.__runChanVese,
			'MCV': self.__runMorphChanVese,
			'AC': self.__runMorphGeodesicActiveContour,
			'FD': self.__runFlood,
			'FF': self.__runFloodFill
			### Add the new algorithm here.
		}

		func = switcher.get(self.params.getAlgo(), "Invalid code") # will take long time
		#These algrotihms only give masks, so are not usable 
		#in the fitness function

		# if self.params.getAlgo() in AlgoHelp().needMask():##
		# 	return FileClass.convertMask(func()) # takes a long time due to func()##
		
			#return self.func()
		return func()



