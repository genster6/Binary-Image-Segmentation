#Strongly coupled with AlgorithmParams and AlgorithmHelper
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

#TODO: Add color segmetation.  
import copy
from collections import OrderedDict 

import numpy as np
import skimage
from skimage import segmentation

#List of all algorithms
algorithmspace = dict()

def algoFromParams(individual):
    '''Converts a param list to an algorithm Assumes order 
    defined in the segmentor class'''
    if (individual[0] in algorithmspace):
        algorithm = algorithmspace[individual[0]]
        return algorithm(individual)
    else:
        raise ValueError("Algorithm not avaliable")

class segmentor(object):
    algorithm = ''

    descriptions = OrderedDict()
    params = OrderedDict()
    descriptions['algorithm'] = 'string code for the algorithm'
    params['algorithm'] = 'None'
    descriptions['beta'] = 'A parameter for randomWalker So, I should take this out'
    params['beta'] = 0.0
    descriptions['tolerance'] = 'A parameter for flood and flood_fill'
    params['tolerance'] = 0.0
    descriptions['scale'] = 'A parameter for felzenszwalb'
    params['scale'] = 0.0
    descriptions['sigma'] = 'sigma value. A parameter for felzenswalb, inverse_guassian_gradient, slic, and quickshift'
    params['sigma'] = 0.0
    descriptions['min_size'] = 'parameter for felzenszwalb'
    params['min_size'] = 0.0
    descriptions['n_segments'] = 'A parameter for slic'
    params['n_segments'] = 0.0
    descriptions['iterations'] = 'A parameter for both morphological algorithms'
    params['iterations'] = 0.0
    descriptions['ratio'] = 'A parameter for ratio'
    params['ratio'] = 0.0
    descriptions['kernel_size'] = 'A parameter for kernel_size'
    params['kernel_size'] = 0.0
    descriptions['max_dist'] = 'A parameter for quickshift'
    params['max_dist'] = 0.0
    descriptions['seed'] = 'A parameter for quickshift, and perhaps other random stuff'
    params['seed'] = 0.0
    descriptions['connectivity'] = 'A parameter for flood and floodfill'
    params['connectivity'] = 0.0
    descriptions['compactness'] = 'A parameter for slic and watershed'
    params['compactness'] = 0.0
    descriptions['mu'] = 'A parameter for chan_vese'
    params['mu'] = 0.0
    descriptions['lambda'] = 'A parameter for chan_vese and morphological_chan_vese'
    params['lambda'] = 0.0
    descriptions['dt'] = '#An algorithm for chan_vese May want to make seperate level sets for different functions e.g. Morph_chan_vese vs morph_geo_active_contour'
    params['dt'] = 0.0
    descriptions['init_level_set_chan'] = 'A parameter for chan_vese and morphological_chan_vese'
    params['init_level_set_chan'] = 0.0
    descriptions['init_level_set_morph'] = 'A parameter for morphological_chan_vese'
    params['init_level_set_morph'] = 'checkerboard'
    descriptions['smoothing'] = 'A parameter used in morphological_geodesic_active_contour'
    params['smoothing'] = 0.0
    descriptions['alpha'] = 'A parameter for inverse_guassian_gradient'
    params['alpha'] = 0.0
    descriptions['balloon'] = 'A parameter for morphological_geodesic_active_contour'
    params['balloon'] = 0.0
    descriptions['seed_pointX'] = 'A parameter for flood and flood_fill'
    params['seed_pointX'] = 0.0
    descriptions['seed_pointY'] = '??'
    params['seed_pointY'] = 0.0
    descriptions['seed_pointZ'] = '??'
    params['seed_pointZ'] = 0.0
    
    keys = list(params.keys())
    paramindexes = list(params.keys())
    
    def __init__(self, paramlist = None):
        if (paramlist):
            self.parse_params(paramlist)

    def parse_params(self, individual):
        print("parsing Paramiters")
        for index, key in enumerate(self.keys):
            #print(f"{index} {key} - {individual[index]}")
            self.params[key] = individual[index]        
    
    def evaluate(self, im):
        return np.zeros(im.shape[0:1])
    
    def paramlist(self):
        plist = []
        for key in self.params:
            plist.append(self.params[key])
        return plist
    
    def __str__(self):
        mystring = f"{self.params['algorithm']} -- "
        for p in self.paramindexes:
            mystring += f"{mystring} {p} = {self.params[p]}\n"
        return mystring
    
class Felzenszwalb(segmentor):
    '''
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
    '''
   
    def __doc__(self):
        myhelp = "Wrapper function for the scikit-image Felzenszwalb segmentor:"
        myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
        return myhelp
    
    def __init__(self, paramlist=None):
        super(Felzenszwalb, self).__init__(paramlist)
        self.params['algorithm'] = 'FB'
        self.params['scale'] = 0.5
        self.params['sigma']= 0.4   
        self.params['min_size'] = 10
        self.params['channel'] = 1
        self.paramindexes = ['scale', 'sigma', 'min_size', 'channel']
        
    def evaluate(self, img):
        output = skimage.segmentation.felzenszwalb(
            img, self.params['scale'], 
            self.params['sigma'], 
            self.params['min_size'],
            multichannel=self.params['channel'])
        return output
algorithmspace['FB'] = Felzenszwalb

class Slic(segmentor):
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
    
    def __init__(self, paramlist=None):
        super(Slic, self).__init__(paramlist)
        self.params['algorithm'] = 'SC'
        self.params['segments'] = 2
        self.params['compact'] = 0.5
        self.params['iterations']= 5   
        self.params['sigma'] = 0.4
        self.params['channel'] = 1
        self.paramindexes = ['segments', 'compact', 'iterations', 'sigma', 'channel']
        
 
    def evaluate(self, img):
        output = skimage.segmentation.slic(img,
                                           n_segments=self.params['segments'], 
                                           compactness=self.params['compact'], 
                                           max_iter=self.params['iterations'],
                                           sigma=self.params['sigma'],
                                           multichannel=self.params['channel'])
        return output
algorithmspace['SC'] = Slic
    

class QuickShift(segmentor):
    '''
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
    '''
    
    def __init__(self, paramlist=None):
        super(QuickShift, self).__init__(paramlist)
        self.params['algorithm'] = 'QS'
        self.params['kernel'] = 2
        self.params['max_dist'] = 20
        self.params['sigma'] = 0.4
        self.params['seed'] = 20
        self.paramindexes = ['kernel', 'max_dist', 'sigma', 'seed']

    def evaluate(self, img):
        output = skimage.segmentation.quickshift(
            img, 
            ratio=self.params['ratio'], 
            kernel_size=self.params['kernel'], 
            max_dist=self.params['max_dist'],
            sigma=self.params['sigma'],
            random_seed=self.params['seed'])
        return output    
algorithmspace['QS'] = QuickShift    
    
class Watershed(segmentor):
    '''
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
    '''
    #Not using connectivity, markers, or offset params as arrays would
    #expand the search space too much.
    #abbreviation for algorithm = WS
    
    def __init__(self, paramlist=None):
        super(Watershed, self).__init__(paramlist)
        self.params['algorithm'] = 'WS'
        self.params['compact'] = 2.0
        self.paramindexes = ['compact']

    def evaluate(self, img):
        output = skimage.segmentation.watershed(
            img,markers=None,
            compactness=self.params['compact'])
        return output
algorithmspace['WS'] = Watershed    


class Chan_Vese(segmentor):
    '''
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
    '''
    #Abbreviation for Algorithm = CV
    def __init__(self, paramlist=None):
        super(Chan_Vese, self).__init__(paramlist)
        self.params['algorithm'] = 'CV'
        self.params['mu'] = 2.0
        self.params['Lambda'] = (10, 20)
        self.params['iterations'] = 10
        self.params['dt'] = 0.10
        self.params['init_level_set_chan'] = 'checkerboard'
        self.paramindexes = ['mu', 'Lambda', 'iterations', 'dt', 'init_level_set_chan']
        
    def evaluate(self, img):
        if(len(img.shape) == 3):
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.chan_vese(
            img, mu=self.params['mu'],
            lambda1=self.params['Lambda'][0], 
            lambda2=self.params['Lambda'][1],
            tol=self.params['tolerance'],
            max_iter=self.params['iterations'], 
            dt=self.params['dt'])
        return output
algorithmspace['CV'] = Chan_Vese 



class Morphological_Chan_Vese(segmentor):
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

    def __init__(self, paramlist=None):
        super(Morphological_Chan_Vese, self).__init__(paramlist)
        self.params['algorithm'] = 'MCV'
        self.params['iterations'] = 10
        self.params['init_level_set_chan'] = 'checkerboard'
        self.params['smoothing'] = 10
        self.params['Lambda'] = (10, 20)
        self.paramindexes = ['iterations','init_level_set_chan', 'smoothing', 'Lambda']
        
    def evaluate(self, img):
        if(len(img.shape) == 3):
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.morphological_chan_vese(
            img, 
            iterations=self.params['iterations'],
            init_level_set=	self.params['init_level_set_chan'],
            smoothing=self.params['smoothing'],
            lambda1=self.params['Lambda'][0], 
            lambda2=self.params['Lambda'][1])
        return output
algorithmspace['MCV'] = Morphological_Chan_Vese 
            
# TODO: Figure out the mask part?
# class RandomWalker(segmentor):
#     algorithm = 'RW'
#     paramindexes = [1, 2]
    
#     def __doc__(self):
#         myhelp = "Wrapper function for the scikit-image random_walker segmentor:"
#         myhelp += f" xx {skimage.segmentation.random_walker.__doc__}"
#         return myhelp
    
#     def __init__(self, beta = 0.5, tolerance = 0.4):
#         self.beta = beta
#         self.tolerance = tolerance
    
#     def evaluate(self, img):
#         #Let's deterime what mode to use
#         mode = "bf"
#         if len(img) < 512 :
#             mode = "cg_mg"

#         #If data is 2D, then this is a grayscale, so multichannel is 
#         output = skimage.segmentation.random_walker(
#             img, labels=mask,
#             beta=self.beta, 
#             tol=self.tolerance, copy=True, 
#             multichannel=True, return_full_prob=False) 
#         return output

        