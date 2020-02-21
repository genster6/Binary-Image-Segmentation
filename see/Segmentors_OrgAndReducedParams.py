""" Segmentor library designed to learn how to segment images using GAs.
This libary actually does not incode the GA itself, instead it just defines
the search parameters the evaluation funtions and the fitness function (comming soon)
"""

# TODO: Research project-clean up the parameters class to reduce the search space
# TODO: Change the seed from a number to a fraction 0-1 which is scaled to image rows and columns
# TODO: Enumerate teh word based measures.

from collections import OrderedDict
import sys

import numpy as np
import skimage
from skimage import segmentation
from skimage import color
from PIL import Image
import pandas as pd  # used in fitness? Can it be removed?
import logging

# List of all algorithms
algorithmspace = dict()

def runAlgo(img, groundImg, individual, returnMask=False):
    logging.getLogger().info(f"Running Algorithm {individual[0]}")
    # img = copy.deepcopy(copyImg)
    seg = algoFromParams(individual)
    mask = seg.evaluate(img)
    logging.getLogger().info("Calculating Fitness")
    fitness = FitnessFunction(mask, groundImg)
    if returnMask:
        return [fitness, mask]
    else:
        return fitness


def algoFromParams(individual):
    """Converts a param list to an algorithm Assumes order 
    defined in the parameters class"""
    if individual[0] in algorithmspace:
        algorithm = algorithmspace[individual[0]]
        return algorithm(individual)
    else:
        raise ValueError("Algorithm not avaliable")


class parameters(OrderedDict):
    descriptions = dict()
    ranges = dict()
    pkeys = []

    ranges["algorithm"] = "['CT','FB','SC','WS','CV','MCV','AC']"
    descriptions["algorithm"] = "string code for the algorithm"

    descriptions["tolerance"] = "A parameter for tolerance and indexing init_level_set_morph"
    ranges["tolerance"] = "[i for i in range(0, 4)]"

    descriptions["scale"] = "A parameter for scale, max_dist, and alpha"
    ranges["scale"] = "[i for i in range(0,10000)]"

    descriptions["sigma"] = "A parameter for sigma"
    ranges["sigma"] = "[float(i)/100 for i in range(0,10,1)]"

    descriptions["min_size"] = "A parameter for min_size, n_segments, and kernel_size"
    ranges["min_size"] = "[i for i in range(0,10000)]"

    descriptions["compactness"] = "A parameter for indexing compactness and lambda"
    ranges["compactness"] = "[i for i in range(0, 9)]"

    descriptions["mu"] = "A parameter for mu, ratio, and balloon"
    ranges["mu"] = "[float(i)/100 for i in range(0,100)]"

    descriptions["smoothing"] = "A parameter smoothing and dt"
    ranges["smoothing"] = "[i for i in range(1, 10)]"


    #     Try to set defaults only once.
    #     Current method may cause all kinds of weird problems.
    #     @staticmethod
    #     def __Set_Defaults__()

    def __init__(self):
        self["algorithm"] = "None"
        self["tolerance"] = 0.0
        self["scale"] = 0.0
        self["sigma"] = 0.0
        self["min_size"] = 0.0
        self["compactness"] = 0.0
        self["mu"] = 0.0
        self["smoothing"] = 0.0
        self.pkeys = list(self.keys())

    def printparam(self, key):
        return f"{key}={self[key]}\n\t{self.descriptions[key]}\n\t{self.ranges[key]}\n"

    def __str__(self):
        out = ""
        for index, k in enumerate(self.pkeys):
            out += f"{index} " + self.printparam(k)
        return out

    def tolist(self):
        plist = []
        for key in self.pkeys:
            plist.append(self.params[key])
        return plist

    def fromlist(self, individual):
        logging.getLogger().info(f"Parsing Parameter List for {len(individual)} parameters")
        for index, key in enumerate(self.pkeys):
            self[key] = individual[index]


class segmentor(object):
    algorithm = ""

    def __init__(self, paramlist=None):
        self.params = parameters()
        if paramlist:
            self.params.fromlist(paramlist)

    def evaluate(self, im):
        return np.zeros(im.shape[0:1])

    def __str__(self):
        mystring = f"{self.params['algorithm']} -- \n"
        for p in self.paramindexes:
            mystring += f"\t{p} = {self.params[p]}\n"
        return mystring


class ColorThreshold(segmentor):
    def __init__(self, paramlist=None):
        super(ColorThreshold, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "CT"
            self.params["mu"] = 0.4
            self.params["sigma"] = 0.6
        self.paramindexes = ["sigma", "mu"]

    def evaluate(self, img): #XX
        channel_num = 1  # TODO: Need to make this a searchable parameter.
        if len(img.shape) > 2:
            if channel_num < img.shape[2]:
                channel = img[:, :, channel_num]
            else:
                channel = img[:, :, 0]
        else:
            channel = img
        pscale = np.max(channel)
        mx = self.params["sigma"] * pscale
        mn = self.params["mu"] * pscale
        if mx < mn:
            temp = mx
            mx = mn
            mn = temp

        output = np.ones(channel.shape)
        output[channel < mn] = 0
        output[channel > mx] = 0

        return output

algorithmspace["CT"] = ColorThreshold


class TripleA (segmentor):
    def __init__(self, paramlist=None):
        super(TripleA, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "AAA"
            self.params["mu"] = 0.4
            self.params["sigma"] = 0.6
        self.paramindexes = ["sigma", "mu"]

    def evaluate(self, img): #XX
        channel_num = 1  # TODO: Need to make this a searchable parameter.
        if len(img.shape) > 2:
            if channel_num < img.shape[2]:
                channel = img[:, :, channel_num]
            else:
                channel = img[:, :, 0]
        else:
            channel = img
        pscale = np.max(channel)
        mx = self.params["sigma"] * pscale
        mn = self.params["mu"] * pscale
        if mx < mn:
            temp = mx
            mx = mn
            mn = temp

        output = np.ones(channel.shape)
        output[channel < mn] = 0
        output[channel > mx] = 0

        return output

algorithmspace["AAA"] = TripleA


class Felzenszwalb(segmentor):
    """
    #felzenszwalb
    #ONLY WORKS FOR RGB
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
    The felzenszwalb algorithms computes a graph based on the segmentation
    Produces an oversegmentation of the multichannel using min-span tree.
    Returns an integer mask indicating the segment labels

    #Variables
    scale: float, higher meanse larger clusters
    sigma: float, std. dev of Gaussian kernel for preprocessing
    min_size: int, minimum component size. For postprocessing
    mulitchannel: bool, Whether the image is 2D or 3D. 2D images
    are not supported at all
    """
    def __doc__(self):
        myhelp = "Wrapper function for the scikit-image Felzenszwalb segmentor:"
        myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
        return myhelp

    def __init__(self, paramlist=None):
        super(Felzenszwalb, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "FB"
            self.params["scale"] = 984
            self.params["sigma"] = 0.09
            self.params["min_size"] = 92
        self.paramindexes = ["scale", "sigma", "min_size"]

    def evaluate(self, img):
        multichannel = False
        if len(img.shape) > 2:
            multichannel = True
        output = skimage.segmentation.felzenszwalb(
            img,
            self.params["scale"],
            self.params["sigma"],
            self.params["min_size"],
            multichannel=True,
        )
        return output

algorithmspace["FB"] = Felzenszwalb


class Slic(segmentor):
    """
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
    """

    def __init__(self, paramlist=None):
        super(Slic, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "SC"
            self.params["min_size"] = 5
            self.params["compactness"] = 5
            self.params["sigma"] = 5
        self.paramindexes = ["min_size", "compactness", "sigma"]

    def evaluate(self, img):
        compactness_list = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        multichannel = False
        if len(img.shape) > 2:
            multichannel = True
        output = skimage.segmentation.slic(
            img,
            n_segments=self.params["min_size"] + 2,
            compactness=compactness_list[self.params["compactness"]],
            max_iter=10,
            sigma=self.params["sigma"],
            convert2lab=True,
            multichannel=multichannel,
        )
        return output

algorithmspace["SC"] = Slic


class QuickShift(segmentor):
    """
    #quickshift
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
    Segments images with quickshift clustering in Color (x,y) space
    #Returns ndarray segmentation mask of the labels
    #Variables
    image -- ndarray, input image
    ratio -- float, balances color-space proximity & image-space  proximity. Higher vals give more weight to color-space
    kernel_size: float, Width of Guassian kernel using smoothing. Higher means fewer clusters
    max_dist -- float: Cut-off point for data distances. Higher means fewer clusters
    return_tree -- bool: Whether to return the full segmentation hierachy tree and distances. Set as False
    sigma -- float: Width of Guassian smoothing as preprocessing.Zero means no smoothing
    convert2lab -- bool: leave alone
    random_seed -- int, Random seed used for breacking ties. 
    """

    def __init__(self, paramlist=None):
        super(QuickShift, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "QS"
            self.params["min_size"] = 5
            self.params["scale"] = 60
            self.params["sigma"] = 5
            self.params["mu"] = 1
        self.paramindexes = ["min_size", "scale", "sigma", "mu"]

    def evaluate(self, img):
        output = skimage.segmentation.quickshift(
            img,
            ratio=self.params["mu"],
            kernel_size=self.params["min_size"],
            max_dist=self.params["scale"],
            sigma=self.params["sigma"],
            random_seed=134,
        )
        return output

algorithmspace["QS"] = QuickShift


class Watershed(segmentor):
    """
    #Watershed
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
    Uses user-markers. treats markers as basins and 'floods' them.
    Especially good if overlapping objects. 
    #Returns a labeled image ndarray
    #Variables
    image -> ndarray, input array
    markers -> int, or int ndarray same shape as image: markers indicating 'basins'
    connectivity -> ndarray, indicates neighbors for connection
    offset -> array, same shape as image: offset of the connectivity
    mask -> ndarray of bools (or 0s and 1s): 
    compactness -> float, compactness of the basins Higher values make more regularly-shaped basin
    """

    # Not using connectivity, markers, or offset params as arrays would
    # expand the search space too much.
    # abbreviation for algorithm = WS

    def __init__(self, paramlist=None):
        super(Watershed, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "WS"
            self.params["compactness"] = 2
        self.paramindexes = ["compactness"]

    def evaluate(self, img):
        compactness_list = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        output = skimage.segmentation.watershed(
            img, markers=None, compactness=compactness_list[self.params["compactness"]]
        )
        return output

algorithmspace["WS"] = Watershed


class Chan_Vese(segmentor):
    """
    #chan_vese
    #ONLY GRAYSCALE
    https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_chan_vese.html
    Segments objects without clear boundaries
    #Returns: segmentation array of algorithm. Optional: When the algorithm converges
    #Variables
    image -> ndarray grayscale image to be segmented
    mu -> float, 'edge length' weight parameter. Higher mu vals make a 'round edge' closer to zero will detect smaller objects
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
        'small disk': Slowest convergence, more likely to find implicit edges
        can also be ndarray same shape as image
    extended_output: bool, If true, adds more returns 
    (Final level set & energies)
    """

    # Abbreviation for Algorithm = CV

    def __init__(self, paramlist=None):
        super(Chan_Vese, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "CV"
            self.params["mu"] = 2.0
            self.params["compactness"] = (10, 20)
            self.params["smoothing"] = 0.10
        self.paramindexes = ["mu", "compactness", "smoothing"]

    def evaluate(self, img):
        lambdas = [(1,1), (1,2), (2,1)]
        tolerance_list = [0.001, 0.01, 0.1, 1]
        if len(img.shape) == 3:
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.chan_vese(
            img,
            mu=self.params["mu"],
            lambda1=lambdas[self.params["compactness"] % 3][0],
            lambda2=lambdas[self.params["compactness"] % 3][1],
            tol=tolerance_list[self.params["tolerance"]],
            max_iter=10,
            dt=self.params["smoothing"] / 10,
        )
        return output

algorithmspace["CV"] = Chan_Vese


class Morphological_Chan_Vese(segmentor):
    """
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
    """

    # Abbreviation for algorithm = MCV

    def __init__(self, paramlist=None):
        super(Morphological_Chan_Vese, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "MCV"
            self.params["tolerance"] = "checkerboard"
            self.params["smoothing"] = 10
            self.params["compactness"] = (10, 20)
        self.paramindexes = ["tolerance", "smoothing", "compactness"]

    def evaluate(self, img):
        init_level_set_morph = ['checkerboard', 'circle']
        lambdas = [(1,1), (1,2), (2,1)]
        if len(img.shape) == 3:
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.morphological_chan_vese(
            img,
            iterations=10,
            init_level_set=init_level_set_morph[self.params['tolerance'] % 2],
            smoothing=self.params["smoothing"],
            lambda1=lambdas[self.params["compactness"] % 3][0],
            lambda2=lambdas[self.params["compactness"] % 3][1],
        )
        return output

algorithmspace["MCV"] = Morphological_Chan_Vese


class MorphGeodesicActiveContour(segmentor):
    """
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
    """

    # Abbrevieation for algorithm = AC

    def __init__(self, paramlist=None):
        super(MorphGeodesicActiveContour, self).__init__(paramlist)
        if not paramlist:
            self.params["algorithm"] = "AC"
            self.params["scale"] = 0.2
            self.params["sigma"] = 0.3
            self.params["tolerance"] = "checkerboard"
            self.params["smoothing"] = 5
            self.params["mu"] = 10
        self.paramindexes = ["scale", "sigma", "tolerance", "smoothing", "mu"]

    def evaluate(self, img):
        init_level_set_morph = ['checkerboard', 'circle']
        # We run the inverse_gaussian_gradient to get the image to use
        gimage = skimage.segmentation.inverse_gaussian_gradient(
            img, self.params["scale"], self.params["sigma"]
        )
        zeros = 0
        output = skimage.segmentation.morphological_geodesic_active_contour(
            gimage,
            10,
            init_level_set_morph[self.params['tolerance'] % 2],
            smoothing=self.params["smoothing"],
            threshold="auto",
            balloon=(self.params["mu"] * 100) - 50,
        )
        return output

algorithmspace["AC"] = MorphGeodesicActiveContour


def countMatches(inferred, groundTruth):
    assert (inferred.shape == groundTruth.shape)    
    m = set()
    n = set()
    setcounts = dict()
    for r in range(inferred.shape[0]):
        for c in range(inferred.shape[1]):
            i_key = inferred[r,c]
            m.add(i_key)
            g_key = groundTruth[r,c]
            n.add(g_key)
            if i_key in setcounts:
                if g_key in setcounts[i_key]:
                    setcounts[i_key][g_key] += 1
                else:
                    setcounts[i_key][g_key] = 1
            else:
                setcounts[i_key] = dict()
                setcounts[i_key][g_key] = 1
    return setcounts, len(m), len(n)


def countsets(setcounts):
    '''
    For each inferred set, find the ground truth set which it maps the most 
    pixels to. So we start from the inferred image, and map towards the 
    ground truth image. For each i_key, the g_key that it maps the most 
    pixels to is considered True. In order to see what ground truth sets
    have a corresponding set(s) in the inferred image, we record these "true" g_keys. 
    This number of true g_keys is the value for L in our fitness function.
    '''
    p = 0
    #L = len(setcounts)
    
    total = 0
    Lsets = set()
    
    best = dict()
    
    for i_key in setcounts: 
        mx = 0
        mx_key = ''
        for g_key in setcounts[i_key]:
            total += setcounts[i_key][g_key] # add to total pixel count
            if setcounts[i_key][g_key] > mx:
                mx = setcounts[i_key][g_key]
                # mx_key = i_key
                mx_key = g_key # record mapping with greatest pixel count
        p += mx
        # Lsets.add(g_key)
        Lsets.add(mx_key) # add the g_key we consider to be correct
        # best[i_key] = g_key
        best[i_key] = mx_key # record "true" mapping
    L = len(Lsets)
    return total-p,L, best



def FitnessFunction(inferred, groundTruth):
    """Takes in two ImageData obects and compares them according to
    skimage's Structual Similarity Index and the mean squared error
    Variables:
    img1 is the validation image
    img2 is an image array segmented by the algorithm.
    imgDim is the number of dimensions of the image.
    """
    # makes sure images are in grayscale
    if len(inferred.shape) > 2:
        logging.getLogger().info("inferred not in grayscale")
        inferred = color.rgb2gray(inferred)
    if len(groundTruth.shape) > 2:  # comment out
        logging.getLogger().info("img2 not in grayscale")
        groundTruth = color.rgb2gray(groundTruth)  # comment out
    
    # Replace with function to output p an L
    # p - number of pixels not correcly mapped
    # L - Number of correctly mapped sets
    setcounts, m, n = countMatches(inferred, groundTruth)
    
    #print(setcounts)
    p, L, best = countsets(setcounts)
    
    logging.getLogger().info(f"p={p}, m={m}, n={n}, L={L}")
    
    error = (p + 2) ** np.log(abs(m - n) + 2)  # / (L >= n)
    # error = (repeat_count + 2)**(abs(m - n)+1)
    # print(f"TESTING - L={L} < n={n} p={p} m={m} error = {error} ")
    if (L < n) or error <= 0 or error == np.inf or error == np.nan:
        logging.warning(
            f"WARNING: Fitness bounds exceeded, using Maxsize - {L} < {n} or {error} <= 0 or {error} == np.inf or {error} == np.nan:"
        )
        error = sys.maxsize
        # print(error)
    return [error, best]
