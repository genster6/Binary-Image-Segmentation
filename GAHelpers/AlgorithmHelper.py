
from deap import algorithms
from deap import base
from deap import tools
from deap import creator
import random

from . import AlgorithmSpace
from .RandomHelp import RandomHelp as RandHelp
import scoop
from scoop import futures
import copy


class AlgoHelp(object) :

	def __init__(self):
		self.indexes = {
				'RW': [1, 2],
				'FB': [3, 4, 5],
				'SC': [6, 7, 8, 4, ],
				'QS': [9, 10, 11, 12],
				'WS': [7],
				'CV': [13, 14, 2, 8, 15, 16],
				'MCV': [8, 17, 18, 14],
				'AC': [20, 4,  18, 19, 21],
				'FD': [22, 13, 2], 
				#Note, index 22 is the seed point. Will have to used diff possible values
				'FF': [22, 13, 2]
				#To add another algorithm, add their indices here
			}
		self.GrayAlgos = {
				'RW',#: Algo.runRandomWalker,
				'WS',#: Algo.runWaterShed,
				'CV',#: Algo.runChanVese,
				'MCV',#: Algo.runMorphChanVese,
				'AC',#: Algo.runMorphGeodesicActiveContour,
				'FD',#: Algo.runFlood,			
				'FF',#: Algo.runFloodFill
			}
		self.RGBAlgos = {
				'RW',#: Algo.runRandomWalker,
				'FB',#: Algo.runFelzenszwalb,
				'SC',#: Algo.runSlic, 
				'QS',#: Algo.runQuickShift,
				'WS',#: Algo.runWaterShed,
				#HAVING SOME TROUBLE WITH AC. NEED TO RETUL
				'AC',#: Algo.runMorphGeodesicActiveContour,
				'FF'#: Algo.runFloodFill
			}
		self.mask = ['FB', 'SC', 'QS', 'CV', 'FD']

		self.usedAlgos = ['FB', 'FF', 'MCV', 'AC', 'CV', 'WS', 'QS']#['FB', 'FF', 'MCV', 'AC', 'FB', 'CV', 'WS', 'QS']

		self.PosVals = [
			copy.deepcopy(self.usedAlgos),
			#Taking out grayscale: CV, MCV, FD
			#Took out  'MCV', 'AC', FB, SC, CV, WS
			#Quickshift(QS) takes a long time, so I'm taking it out for now.
			[i for i in range(0,1000)], #betas
			[float(i)/1000 for i in range(0,1000,1)],  #tolerance
			[i for i in range(0,1000)], #scale
			[float(i)/100 for i in range(0,10,1)], #sigma
			#Sigma should be weighted more from 0-1
			[i for i in range(0,1000)], #min_size
			[i for i in range(2,1000)], #n_segments
			[10], #iterations
			[float(i)/100 for i in range(0,100)], #ratio
			[i for i in range(0,1000)], #kernel
			[i for i in range(0,1000)], #max_dists
			[134], #random_seed
			[i for i in range(0, 9)],  #connectivity
			#How much a turtle likes its neighbors
			
			[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], #compactness
			#I may want to remake compactness with list capabilities
			[float(i)/100 for i in range(0,100)], #mu
			#The values for Lambda1 and Lambda2 respectively
			[[1,1], [1,2], [2,1]], #Lambdas
			[float(i)/10 for i in range(0,100)], #dt
			['checkerboard', 'disk', 'small disk'], #init_level_set_chan_vese
			['checkerboard', 'circle'], #init_level_set_morph
			#Should weight 1-4 higher
			[i for i in range(1, 10)], #smoothing
			[i for i in range(0,1000)], #alphas
			#Should weight values -1, 0 and 1 higher
			[i for i in range(-50,50)], #balloon
			#Getting the seedpoint for floodfill

			#ADD ADDITIONAL VALUES FOR PARAMETERS HERE
			#TODO: Make seed point input parameter
			#Dimensions of the imag

		]
	def allVals(self): return self.PosVals
	#Will return a dictionary of each algorithm by their code and their
	#associdated indices. Specifically used in 'mutate' in GeneticHelp
	#class. Will later be used in 'skimageCrossRandom'. 
	def algoIndexes(self): return self.indexes

	#Grayscale vs Multichannel algorithms
	def channelAlgos(self,img):
		#Depends on the channel
		if (img.getDim() > 2): return self.RGBAlgos
		else: return self.GrayAlgos

	#Algorithms that return the mask of an object
	def needMask(self): return self.mask

	def AlgoSpaceSwitcher(self): return self.allAlgos

	def usedAlgos(self): return self.UsedAlgos

	def makeToolbox(self, population, seedX, seedY, seedZ):

		#Was having circular dependencies
		from .GeneticHelp import GeneticHelp as GA
		#I do know that this is bad practice

		#Minimizing fitness function
		creator.create("FitnessMin", base.Fitness, weights=(-0.000001,))

		creator.create("Individual", list, fitness=creator.FitnessMin)
		
		#The functions that the GA knows
		toolbox = base.Toolbox()
		#Attribute generator
		toolbox.register("attr_bool", random.randint, 0, 1000)
		
		#Genetic functions
		toolbox.register("mate", GA.skimageCrossRandom) #crossover
		toolbox.register("evaluate", GA.runAlgo) #Fitness
		toolbox.register("mutate", GA.mutate) #Mutation
		toolbox.register("select", tools.selTournament, tournsize=5) #Selection
		# toolbox.register("select", tools.selBest)
		toolbox.register("map", futures.map) #So that we can use scoop
		#May want to later do a different selection process
		
		#Here we register all the parameters to the toolbox
		SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT = 0, 1, 0.5	
		#Perhaps weight iterations
		ITER = 10
		SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT = 1, 4, 0.5
		BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT = -1, 1, 0.9

		#We choose the parameters, for the most part, random
		toolbox.register("attr_Algo", random.choice, self.PosVals[0])
		toolbox.register("attr_Beta", random.choice, self.PosVals[1])
		toolbox.register("attr_Tol", random.choice, self.PosVals[2])
		toolbox.register("attr_Scale", random.choice, self.PosVals[3])
		#While sigma can be any positive value, it should be small (0-1). 
		toolbox.register("attr_Sigma", RandHelp.weighted_choice, 
			self.PosVals[4], SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT)
		toolbox.register("attr_minSize", random.choice, self.PosVals[5])
		toolbox.register("attr_nSegment", random.choice, self.PosVals[6])
		toolbox.register("attr_iterations", int, 10)
		toolbox.register("attr_ratio", random.choice, self.PosVals[8])
		toolbox.register("attr_kernel", random.choice, self.PosVals[9])
		toolbox.register("attr_maxDist", random.choice, self.PosVals[10])
		toolbox.register("attr_seed", int, 134) #regular seed is too large
		toolbox.register("attr_connect", random.choice, self.PosVals[12])
		toolbox.register("attr_compact", random.choice, self.PosVals[13])
		toolbox.register("attr_mu", random.choice, self.PosVals[14])
		toolbox.register("attr_lambda", random.choice, self.PosVals[15])
		toolbox.register("attr_dt", random.choice, self.PosVals[16])
		toolbox.register("attr_init_chan", random.choice, 
			self.PosVals[17])
		toolbox.register("attr_init_morph", random.choice, 
			self.PosVals[18])
		#smoothing should be 1-4, but can be any positive number
		toolbox.register("attr_smooth", RandHelp.weighted_choice,
			self.PosVals[19], SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT)
		toolbox.register("attr_alphas", random.choice, self.PosVals[20])
		#Should be from -1 to 1, but can be any value
		toolbox.register("attr_balloon", RandHelp.weighted_choice, 
			self.PosVals[21], BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT)
		
		#Need to register a random seed_point
		toolbox.register("attr_seed_pointX", random.choice, seedX)
		toolbox.register("attr_seed_pointY", random.choice, seedY)
		toolbox.register("attr_seed_pointZ", random.choice, seedZ)

		#REGISTER MORE PARAMETERS TO THE TOOLBOX HERE
		#FORMAT:
		#toolbox.register("attr_param", random.choice, param_list)

		#Container: data type
		#func_seq: List of function objects to be called in order to fill 
		#container
		#n: number of times to iterate through list of functions
		#Returns: An instance of the container filled with data returned 
		#from functions
		func_seq = [toolbox.attr_Algo, toolbox.attr_Beta, toolbox.attr_Tol,
			toolbox.attr_Scale, toolbox.attr_Sigma, toolbox.attr_minSize,
			toolbox.attr_nSegment, 
			toolbox.attr_iterations, toolbox.attr_ratio,
			toolbox.attr_kernel, toolbox.attr_maxDist, toolbox.attr_seed, 
			toolbox.attr_connect, toolbox.attr_compact, toolbox.attr_mu, 
			toolbox.attr_lambda, toolbox.attr_dt, toolbox.attr_init_chan,
			toolbox.attr_init_morph, toolbox.attr_smooth, 
			toolbox.attr_alphas, toolbox.attr_balloon, 
			toolbox.attr_seed_pointX, toolbox.attr_seed_pointY,
			toolbox.attr_seed_pointZ]
		#AT THE END OF THE 'func_seq' ADD MORE PARAMETERS
		#print(func_seq)
		#Here we populate our individual with all of the parameters
		toolbox.register("individual", tools.initCycle, creator.Individual
			, func_seq, n=1)


		#And we make our population
		toolbox.register("population", tools.initRepeat, list, 
			toolbox.individual, n=population)

		
		#def makeToolBox(self):
		return toolbox


