import matplotlib.pylab as plt

import re

import imageio
import numpy as np

from see import JupyterGUI
import RunCoco
from see import Segmentors
import inspect
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the see-Semgent algorithm')
    parser.add_argument("-o", "--outputfolder", 
                        help="Output Folder", 
                        type=str, default="./output/")
    
    #Parsing Inputs
    args = parser.parse_args()
    print(args)

    outputfolder = args.outputfolder

if  not 'outputfolder' in locals():
    outputfolder = './output/'

images, masks, outputs = RunCoco.getCocoFolderLists(outputfolder)

reportfile = open(f'{outputfolder}report.md','w')
reportfile.write("# Search Report\n\n")

#Fitness for all files
file = open(outputfolder+'params.txt','r')
fitness = []
filelines = file.readlines()
params = filelines[0]
reportfile.write(params)
reportfile.write("\n\n")
for line in filelines[1:]:
    match = re.search(' ', line)
    fitness.append(float(line[:match.start()]))


reportfile.write("```python\n")
seg = Segmentors.algoFromParams(eval(params))
print(inspect.getsource(seg.evaluate))
reportfile.write(inspect.getsource(seg.evaluate))
reportfile.write("```\n")


fitness = np.array(fitness)
total = len(fitness)
sorted_fitness = fitness.copy()
sorted_fitness.sort()


fivehundred_err = len(fitness[fitness < 500]) / total * 100
reportfile.write(f"This parameter got {fivehundred_err}% on all the Coco files\n")


plt.hist(fitness[fitness < 500], 50);
plt.xlabel('fitness values')
plt.ylabel('image count')
figurename = 'histogram.png'
plt.savefig(outputfolder+figurename)
reportfile.write(f"![histogram of fitnesses < 500]({figurename})\n")

###How fitness changes over time
file = open("testoutput.out","r") 
match = ''
tfitness = []
for line in file.readlines():
    if re.search('#BEST', line):
        match = re.search(" \- \[", line)
        tfitness.append(float(line[8:match.start()]))
file.close()

plt.plot(tfitness)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Evolution of the fitness');
figurename = 'fitnessvstime.png'
plt.savefig(outputfolder+figurename)
reportfile.write(f"![How fitness changed with generation on single image]({figurename})\n")

#for imagename, maskname, outputname in zip(images, masks, output):
index = 10

imagename = images[index]
maskname = masks[index]
outputname = outputs[index]

print("\n")

image = imageio.imread(imagename)
mask = imageio.imread(maskname)
output = imageio.imread(outputname)

JupyterGUI.showthree(image,mask,output)

figurename = 'showthree.png'
plt.savefig(outputfolder+figurename)
reportfile.write(f"![Show original image, ground truth image, and segmented image]({figurename})\n")


reportfile.close()
print(f"report written to {outputfolder}report.md")