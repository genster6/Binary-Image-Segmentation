#Strongly coupled with AlgorithmParams and AlgorithmHelper
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

#TODO: Add color segmetation.  
from collections import OrderedDict 

import numpy as np
import skimage
from skimage import segmentation
from PIL import Image

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

class parameters(OrderedDict):
    descriptions = dict()
    ranges = dict()
    pkeys = []
    
    def __init__(self):
        self['algorithm'] = 'None'
        self.ranges['algorithm'] = "['FB','SC','WS','CV','MCV','AC']"
        self.descriptions['algorithm'] = 'string code for the algorithm'
        
        self.descriptions['beta'] = 'A parameter for randomWalker So, I should take this out'
        self.ranges['beta'] = "[i for i in range(0,10000)]"
        self['beta'] = 0.0

        self.descriptions['tolerance'] = 'A parameter for flood and flood_fill'
        self.ranges['tolerance'] = "[float(i)/1000 for i in range(0,1000,1)]"
        self['tolerance'] = 0.0

        self.descriptions['scale'] = 'A parameter for felzenszwalb'
        self.ranges['scale'] = "[i for i in range(0,10000)]"
        self['scale'] = 0.0

        self.descriptions['sigma'] = 'sigma value. A parameter for felzenswalb, inverse_guassian_gradient, slic, and quickshift'
        self.ranges['sigma'] = "[float(i)/100 for i in range(0,10,1)]"
        self['sigma'] = 0.0
        
        self.descriptions['min_size'] = 'parameter for felzenszwalb'
        self.ranges['min_size'] = "[i for i in range(0,10000)]"
        self['min_size'] = 0.0
        
        self.descriptions['n_segments'] = 'A parameter for slic'
        self.ranges['n_segments'] = "[i for i in range(2,10000)]"
        self['n_segments'] = 0.0
        
        self.descriptions['iterations'] = 'A parameter for both morphological algorithms'
        self.ranges['iterations'] = "[10, 10]"
        self['iterations'] = 10
        
        self.descriptions['ratio'] = 'A parameter for ratio'
        self.ranges['ratio'] = "[float(i)/100 for i in range(0,100)]"
        self['ratio'] = 0.0
        
        self.descriptions['kernel_size'] = 'A parameter for kernel_size'
        self.ranges['kernel_size'] = "[i for i in range(0,10000)]"
        self['kernel_size'] = 0.0
        
        self.descriptions['max_dist'] = 'A parameter for quickshift'
        self.ranges['max_dist'] = "[i for i in range(0,10000)]"
        self['max_dist'] = 0.0
        
        self.descriptions['seed'] = 'A parameter for quickshift, and perhaps other random stuff'
        self.ranges['seed'] = "[134]"
        self['seed'] = 0.0
        
        self.descriptions['connectivity'] = 'A parameter for flood and floodfill'
        self.ranges['connectivity'] = "[i for i in range(0, 9)]"
        self['connectivity'] = 0.0
        
        self.descriptions['compactness'] = 'A parameter for slic and watershed'
        self.ranges['compactness'] = "[0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]"
        self['compactness'] = 0.0
        
        self.descriptions['mu'] = 'A parameter for chan_vese'
        self.ranges['mu'] = "[float(i)/100 for i in range(0,100)]"
        self['mu'] = 0.0

        self.descriptions['lambda'] = 'A parameter for chan_vese and morphological_chan_vese'
        self.ranges['lambda'] = [[1,1], [1,2], [2,1]]
        self['lambda'] = (1,1)

        self.descriptions['dt'] = '#An algorithm for chan_vese May want to make seperate level sets for different functions e.g. Morph_chan_vese vs morph_geo_active_contour'
        self.ranges['dt'] = "[float(i)/10 for i in range(0,100)]"
        self['dt'] = 0.0

        self.descriptions['init_level_set_chan'] = 'A parameter for chan_vese and morphological_chan_vese'
        self.ranges['init_level_set_chan'] = "['checkerboard', 'disk', 'small disk']"
        self['init_level_set_chan'] = 0.0

        self.descriptions['init_level_set_morph'] = 'A parameter for morphological_chan_vese'
        self.ranges['init_level_set_morph'] = "['checkerboard', 'circle']"
        self['init_level_set_morph'] = 'checkerboard'

        self.descriptions['smoothing'] = 'A parameter used in morphological_geodesic_active_contour'
        self.ranges['smoothing'] = [i for i in range(1, 10)]
        self['smoothing'] = 0.0
        
        self.descriptions['alpha'] = 'A parameter for inverse_guassian_gradient'
        self.ranges['alpha'] = "[i for i in range(0,10000)]"
        self['alpha'] = 0.0
        
        self.descriptions['balloon'] = 'A parameter for morphological_geodesic_active_contour'
        self.ranges['balloon'] = "[i for i in range(-50,50)]"
        self['balloon'] = 0.0
        
        self.descriptions['seed_pointX'] = 'A parameter for flood and flood_fill'
        self.ranges['seed_pointX'] = "[0.0]"
        self['seed_pointX'] = 0.0
        
        self.descriptions['seed_pointY'] = '??'
        self.ranges['seed_pointY'] = "[0.0]"
        self['seed_pointY'] = 0.0
        
        self.descriptions['seed_pointZ'] = '??'
        self.ranges['seed_pointZ'] = "[0.0]"
        self['seed_pointZ'] = 0.0
        
        self.pkeys = list(self.keys())
        
    def printparam(self, key):
        return f"{key}={self[key]}\n\t{self.descriptions[key]}\n\t{self.ranges[key]}\n"

    def __str__(self):
        out = ""
        for k in self.pkeys:
            out += self.printparam(k)
        return out
        
    def tolist(self):
        plist = []
        for key in pkeys:
            plist.append(self.params[key])
        return plist
    
    def fromlist(self, individual):
        print("Parsing Parameter List")
        for index, key in enumerate(self):
            self[key] = individual[index]       
        
class segmentor(object):
    algorithm = ''

    params = parameters()
    
    def __init__(self, paramlist = None):
        if (paramlist):
            self.params.fromlist(paramlist)      
    
    def evaluate(self, im):
        return np.zeros(im.shape[0:1])
    
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
        self.paramindexes = ['scale', 'sigma', 'min_size', 'channel']
        
    def evaluate(self, img):
        multichannel = False
        if (len(img.shape) > 2):
            multichannel = True
        output = skimage.segmentation.felzenszwalb(
            img, self.params['scale'], 
            self.params['sigma'], 
            self.params['min_size'],
            multichannel=True)
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
        self.params['n_segments'] = 2
        self.params['compactness'] = 0.5
        self.params['iterations']= 5   
        self.params['sigma'] = 0.4
        self.paramindexes = ['n_segments', 'compactness', 'iterations', 'sigma']
        
 
    def evaluate(self, img):
        multichannel = False
        if (len(img.shape) > 2):
            multichannel = True
        output = skimage.segmentation.slic(
            img,
            n_segments=self.params['n_segments'], 
            compactness=self.params['compactness'], 
            max_iter=self.params['iterations'],
            sigma=self.params['sigma'],
            multichannel=multichannel)
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
        self.params['kernel_size'] = 2
        self.params['max_dist'] = 20
        self.params['sigma'] = 0.4
        self.params['seed'] = 20
        self.paramindexes = ['kernel_size', 'max_dist', 'sigma', 'seed']

    def evaluate(self, img):
        output = skimage.segmentation.quickshift(
            img, 
            ratio=self.params['ratio'], 
            kernel_size=self.params['kernel_size'], 
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
        self.params['compactness'] = 2.0
        self.paramindexes = ['compactness']

    def evaluate(self, img):
        output = skimage.segmentation.watershed(
            img,markers=None,
            compactness=self.params['compactness'])
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
        self.params['lambda'] = (10, 20)
        self.params['iterations'] = 10
        self.params['dt'] = 0.10
        self.params['init_level_set_chan'] = 'checkerboard'
        self.paramindexes = ['mu', 'lambda', 'iterations', 'dt', 'init_level_set_chan']
        
    def evaluate(self, img):
        if(len(img.shape) == 3):
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.chan_vese(
            img, mu=self.params['mu'],
            lambda1=self.params['lambda'][0], 
            lambda2=self.params['lambda'][1],
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
        self.params['lambda'] = (10, 20)
        self.paramindexes = ['iterations','init_level_set_chan', 'smoothing', 'lambda']
        
    def evaluate(self, img):
        if(len(img.shape) == 3):
            img = skimage.color.rgb2gray(img)
        output = skimage.segmentation.morphological_chan_vese(
            img, 
            iterations=self.params['iterations'],
            init_level_set=	self.params['init_level_set_chan'],
            smoothing=self.params['smoothing'],
            lambda1=self.params['lambda'][0], 
            lambda2=self.params['lambda'][1])
        return output
algorithmspace['MCV'] = Morphological_Chan_Vese 

class MorphGeodesicActiveContour(segmentor):
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

    def __init__(self, paramlist=None):
        super(MorphGeodesicActiveContour, self).__init__(paramlist)
        self.params['algorithm'] = 'AC'
        self.params['alpha'] = 0.2
        self.params['sigma'] = 0.3
        self.params['iterations'] = 10
        self.params['init_level_set_morph'] = 'checkerboard'
        self.params['smoothing'] = 5
        self.params['balloon'] = 10
        self.paramindexes = ['alpha', 'sigma', 'iterations', 'init_level_set_morph', 'smoothing', 'balloon']

    def evaluate(self, img):
        #We run the inverse_gaussian_gradient to get the image to use
        gimage = skimage.segmentation.inverse_gaussian_gradient(
            img, self.params['alpha'], 
            self.params['sigma'])
        zeros = 0
        output = skimage.segmentation.morphological_geodesic_active_contour(
            gimage, self.params['iterations'], 
            self.params['init_level_set_morph'],
            smoothing= self.params['smoothing'], 
            threshold='auto', 
            balloon=self.params['balloon'])
        return output
algorithmspace['AC'] = MorphGeodesicActiveContour 

# class Flood(segmentor):
#     '''
#     #flood
#     #DOES NOT SUPPORT MULTICHANNEL IMAGES
#     https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
#     Uses a seed point and to fill all connected points within/equal to
#         a tolerance around the seed point
#     #Returns a boolean array with 'flooded' areas being true
#     #Variables
#     image: ndarray, input image
#     seed_point: tuple/int, x,y,z referring to starting point for flood 
#         fill
#     selem: ndarray of 1's and 0's, Used to determine neighborhood of
#         each pixel
#     connectivity: int, Used to find neighborhood of each pixel. Can use 
#         this or selem.
#     tolerance: float or int, If none, adjacent values must be equal to 
#         seed_point. Otherwise, how likely adjacent values are flooded.
#     '''
#     #Abbreviation for algorithm = FD

#     def __init__(self, paramlist=None):
#         super(Flood, self).__init__(paramlist)
#         self.params['algorithm'] = 'AC'
#         self.params['seed_pointX'] = 10
#         self.params['seed_pointY'] = 20
#         self.params['seed_pointZ'] = 0
#         self.params['connect'] = 4
#         self.params['tolerance'] = 0.5
#         self.paramindexes = ['seed', 'connect', 'tolerance']

#     def evaluate(self, img):
#         output = skimage.segmentation.flood(
#             img,
#             (self.params['seed_pointX'], 
#              self.params['seed_pointY'], 
#              self.params['seed_pointZ']),
#             connectivity=self.params['connect'], 
#             tolerance=self.params['tolerance'])
#         return output
# algorithmspace['FD'] = Flood 


# class FloodFill(segmentor):
#     '''
#     #flood_fill
#     https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
#     Like a paint-bucket tool in paint. Like flood, but changes the 
#         color equal to new_type
#     #Returns A filled array of same shape as the image
#     #Variables
#     image: ndarray, input image
#     seed_point: tuple or int, starting point for filling (x,y,z)
#     new_value: new value to set the fill to (e.g. color). Must agree
#         with image type
#     selem: ndarray, Used to find neighborhood of filling
#     connectivity: Also used to find neighborhood of filling if selem is
#         None
#     tolerance: float or int, If none, adjacent values must be equal to 
#         seed_point. Otherwise, how likely adjacent values are flooded.
#     inplace: bool, If true, the flood filling is applied to the image,
#         if False, the image is not modified. Default False, don't 
#         change
#     '''
#     #Abbreviation for algorithm == FF
    
#     def __init__(self, paramlist=None):
#         super(FloodFill, self).__init__(paramlist)
#         self.params['algorithm'] = 'AC'
#         self.params['seed_pointX'] = 10
#         self.params['seed_pointY'] = 20
#         self.params['seed_pointZ'] = 0
#         self.params['connect'] = 4
#         self.params['tolerance'] = 0.5
#         self.paramindexes = ['seed', 'connect', 'tolerance']
        
#     def evaluate(self, img):
#         output = skimage.segmentation.flood_fill(
#             img, 
#             (self.params['seed_pointX'], 
#              self.params['seed_pointY'], 
#              self.params['seed_pointZ']),
#             134,  #TODO: Had coded value
#             connectivity= self.params['connect'], 
#             tolerance=self.params['tolerance'])
#         try:
#             #I'm not sure if this will work on grayscale
#             image = Image.fromarray(output.astype('uint8'), '1')
#         except ValueError:
#             image = Image.fromarray(output.astype('uint8'), 'RGB')

#         width = image.width
#         height = image.width


#         #Converting the background to black
#         for x in range(0, width):
#             for y in range(0, height):
#                 #First check for grayscale
#                 pixel = image.getpixel((x,y))
#                 if pixel[0] == 134:
#                     image.putpixel((x,y), 134)
#                     continue
#                 else:
#                     image.putpixel((x,y), 0)
#                     #print(image.getpixel((x,y)))

#         #image.convert(mode='L')
#         pic = np.array(image)
#         return pic
# algorithmspace['FF'] = FloodFill 

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

        