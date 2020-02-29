#Full HPC Coco Run
import argparse
import random
import logging
import sys
import glob
import os

from skimage import color
import imageio

import see
from see import GeneticSearch
from see import Segmentors


imagefolder = "Image_data/Coco_2017_unlabeled/rgbd_plant/"
maskfolder = "Image_data/Coco_2017_unlabeled/rgbd_new_label"

parser = argparse.ArgumentParser(description='Run the see-Semgent algorithm')
parser.add_argument("-a", "--algorithm", 
                    help="string for algorithm", 
                    type=str, default="")
parser.add_argument("-g", "--generations", 
                    help="Number of Generations to run in search", 
                    type=int, default=2)
parser.add_argument("-s", "--seed", 
                    help="Random Seed", 
                    type=int, default=0)
parser.add_argument("-p", "--pop", 
                    help="Population (file or number)", 
                    type=int, default="10")
parser.add_argument("-i", "--index", 
                    help="Input image index (used for looping)", 
                    type=int, 
                    default=0)
parser.add_argument("-o", "--outputfolder", 
                    help="Output Folder", 
                    type=str, default="./output/")

#Parsing Inputs
args = parser.parse_args()
print(args)

#Setting Log Level
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
#logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#Make List of all files
names = glob.glob(f'{imagefolder}/rgb*.png')
names.sort()

#Create Segmentor
if args.algorithm:
    params = eval(args.algorithm)
else:
    params=''
print(f"Algorithm={params}")
    

if params:
    #Check to see if list of parameters is passed
    if len(params[0])>1:
        #Pick this parameter from list
        if args.index:
            params = params[args.index]
else:
    #Pick out this image and mask
    index = args.index
    name = names[index]
    imagename = os.path.basename(name)
    image_id = imagename[4:-4]
    label = f"label_{image_id}{index}.png"

    # Load this image and mask
    img = imageio.imread(name)
    gmask = imageio.imread(f"{maskfolder}/{label}")

    #Run random Search
    random.seed(args.seed)
    ee = GeneticSearch.Evolver(img, gmask, pop_size=args.pop)
    ee.run(args.generations)# TODO: ADD THIS, checkpoint=args.checkpointfile)
    params = ee.hof[0]
    
#Create segmentor from params
file = open(f"{args.outputfolder}params.txt","w") 
file.write(str(params)) 

seg = Segmentors.algoFromParams(params)

#Loop though images
for index, name in enumerate(names):

    imagename = os.path.basename(name)
    image_id = imagename[4:-4]
    label = f"label_{image_id}{index}.png"

    # Loop over image files
    img = imageio.imread(name)
    gmask = imageio.imread(f"{maskfolder}/{label}")
    if len(gmask.shape) > 2:
        gmask = color.rgb2gray(gmask)

    #Evaluate image
    mask = seg.evaluate(img)

    #Save Mask to output
    imageio.imwrite(f"{args.outputfolder}{label}", mask)

    fitness,_ = Segmentors.FitnessFunction(mask,gmask)
    file.write(f"{fitness} {label} {imagename}")
    print(f"evaluating {imagename} --> {fitness}")
    
file.close() 




