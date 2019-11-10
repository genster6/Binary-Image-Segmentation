#Lightly Coupled with ImageData class
#A class with accessors and modifiers for all the parameters used in
#the skimage algorithms

import random

class AlgorithmParams(object):

    def __str__(self):
        return(f"Algorithm={self.algorithm}")
    


    #Reading in all of the data
    def __init__(self, img, individual):

        #img is an ImageData object
        self.Image = img #The image, usually an ImageData object
        self.algorithm = individual[0] #The string code for the algorithm
        self.beta = individual[1] #A parameter for randomWalker So, I should 
        #take this out
        self.tolerance = individual[2] #A parameter for flood and flood_fill
        self.scale = individual[3] #A parameter for felzenszwalb
        self.sigma = individual[4] #sigma value. A parameter for felzenswalb,
            #inverse_guassian_gradient, slic, and quickshift
        self.min_size = individual[5] #A parameter for felzenszwalb
        self.n_segments = individual[6] #A parameter for slic
        self.iterations = individual[7] #A parameter for both morphological 
        #algorithms
        self.ratio = individual[8] #A parameter for ratio
        self.kernel_size = individual[9] #A parameter for kernel_size
        self.max_dist = individual[10] #A parameter for quickshift
        self.seed = individual[11] #A parameter for quickshift, 
            #and perhaps other random stuff
        self.connectivity = individual[12] #A parameter for flood and 
        #floodfill
        self.compactness = individual[13] #A parameter for slic and 
        #watershed
    
        self.mu = individual[14] #A parameter for chan_vese
        self.lambda1 = individual[15][0] #A parameter for chan_vese and 
        #morphological_chan_vese
        self.lambda2 = individual[15][1] #A parameter for chan_vese and 
        #morphological_chan_vese
        self.dt = individual[16] #An algorithm for chan_vese
        #May want to make seperate level sets for different functions
            #e.g. Morph_chan_vese vs morph_geo_active_contour
        self.init_level_set_chan = individual[17] #A parameter
            #for chan_vese and morphological_chan_vese
        self.init_level_set_morph = individual[18] #A parameter
            #for morphological_chan_vese 
        self.smoothing = individual[19] #A parameter used in
            #morphological_geodesic_active_contour
        self.threshold='auto' #A parameter for 
            #morphological_geodesic_active_contour 
        self.alpha = individual[20] #A parameter for inverse_guassian_gradient
        self.balloon = individual[21] #A parameter for
            #morphological_geodesic_active_contour
        self.seed_pointX = individual[22] #A parameter for flood and 
        #flood_fill
        self.seed_pointY = individual[23]
        self.seed_pointZ = individual[24]


        ### ADD ADDITIONAL PARAMETERS HERE
        #Format:
        #self.new_param = individual[last_num + 1]
        


    #Accessors
    def getImage(self): return self.Image
    def getAlgo(self): return self.algorithm
    def getBeta(self): return self.beta
    def getTolerance(self): return self.tolerance
    def getScale(self): return self.scale
    def getSigma(self): return self.sigma
    def getMinSize(self): return self.min_size
    def getSegments(self): return self.n_segments
    def getCompact(self): return self.compactness
    def getIters(self): return self.iterations
    def getRatio(self): return self.ratio
    def getKernel(self): return self.kernel_size
    def getMaxDist(self): return self.max_dist
    def getSeed(self): return self.seed
    def getConnect(self): return self.connectivity
    def getMu(self): return self.mu
    def getLambdaOne(self): return self.lambda1
    def getLambdaTwo(self): return self.lambda2
    def getDT(self): return self.dt
    def getInitLvlSetChan(self): return self.init_level_set_chan
    def getInitLvlSetMorph(self): return self.init_level_set_morph
    def getSmoothing(self): return self.smoothing
    def getThresh(self): return self.threshold
    def getAlpha(self): return self.alpha
    def getSmoothing(self): return self.smoothing
    def getBalloon(self): return self.balloon
    def getSeedPoint(self): return (self.seed_pointX, self.seed_pointY, self.seed_pointZ)


    #Modifiers
    def changeImage(self, newImg):
        self.Image = newImg
        #If the dimension is different, we should also reset the seed 
        #shape
        if (len(newImg.getShape()) != len(self.seed_point)):
            self.seed_point = [random.randrange(0, dim) for dim in 
            newImg.getShape()]
    def changeAlgo(self, algo):
        self.algorithm = algo
    def changeBeta(self, beta):
        self.beta = beta
    def changeTolerance(self, tol):
        self.tolerance = tol
    def changeScale(self, scale):
        self.scale = scale
    def changeSigma(self, sigma):
        self.sigma = sigma
    def changeMinSize(self, minSize):
        self.min_size = minSize
    def changeSegments(self, segments):
        self.n_segments = segments
    def changeCompact(self, compactness):
        self.compactness = compactness
    def changeIter(self, iters):
        self.iterations = iters
    def changeRatio(self, ratio):
        self.ratio = ratio
    def changeKernel(self, kernel):
        self.kernel_size = kernel
    def changeMaxDist(self, maxDist):
        self.max_dist = maxDist
    def changeSeed(self, random_seed):
        self.seed = random_seed
    def changeConnect(self, connects):
        self.connectivity = connects
    def changeMu(self, mu):
        self.mu = mu
    #Lambda1 and lambda2
    def changeL1(self, lambda1):
        self.lambda1 = lambda1
    def changeL2(self, lambda2):
        self.lambda2 = lambda2
    def changeDT(self, dt):
        self.dt = dt
    def changeInitLvlSetChan(self, lvlSet):
        self.init_level_set_chan = lvlSet
    def changeInitLvlSetMorph(self, lvlSet):
        self.init_level_set_morph = lvlSet
    def changeSmoothing(self, smoothing):
        self.smoothing = smoothing
    def changeThresh(self, threshold):
        self.threshold = threshold
    def changeAlpha(self, alpha):
        self.alpha = alpha
    def changeSmoothing(self, smoothing):
        self.smoothing = smoothing
    def changeBalloon(self, balloon):
        self.balloon = balloon
    def changeSeedPoint(self, seedPt):
        self.seed_pointX = seedPt[0]
        self.seed_pointY = seedPt[1]
        self.seed_pointZ = seedPt[2]
