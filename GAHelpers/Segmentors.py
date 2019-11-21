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
    if (individual[0] in algorithmspace):
        algorithm = algorithmspace[individual[0]]
    return algorithm(individual)

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
    params['init_level_set_morph'] = 0.0
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
        self.params['iters']= 5   
        self.params['sigma'] = 0.4
        self.params['channel'] = 1
        self.paramindexes = ['segments', 'compact', 'iters', 'sigma', 'channel']
        
 
    def evaluate(self, img):
        output = skimage.segmentation.slic(img,
                                           n_segments=self.params['segments'], 
                                           compactness=self.params['compact'], 
                                           max_iter=self.params['iters'],
                                           sigma=self.params['sigma'],
                                           multichannel=self.params['channel'])
        return output
algorithmspace['SC'] = Slic
    
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

        