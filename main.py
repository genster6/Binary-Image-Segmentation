# TODO: Input Seed search space
# TODO: Rewrite code to use "import deap" and deap.base etc. to make easier to read
# TODO: Avoid syntax "from x import y" As this makes the code harder to read and determine dependancies

import time
import random
import os
import sys
import pickle

# TODO: Check each import to ensure it is required in the code below. Remove it necessary
# https://github.com/DEAP/deap
import deap
import cv2

# TODO: Avoid syntax "from x import y" As this makes the code harder to read and determine dependancies
# TODO: I don't like the name GAHelplers. Name implies library will work for all GAs. This should be the segment library We need to come up with a good name.
from GAHelpers import ImageData
from GAHelpers import AlgorithmSpace
from GAHelpers.AlgorithmSpace import AlgorithmSpace
from GAHelpers import AlgorithmParams
from GAHelpers import FileClass
from GAHelpers.FileClass import FileClass
from GAHelpers.AlgorithmHelper import AlgoHelp

class SegmentImage():
    # TODO Add checkpoint name so we can run two runs at once on the hpc.
    # TODO: Make all input params changeable by input arguments
    IMAGE_PATH = 'Image_data/Coco_2017_unlabeled//rgbd_plant'
    #IMAGE_PATH = 'Image_data/sky_data'
    # TODO: Change validation to ground_truth
    GROUNDTRUTH_PATH = 'Image_data/Coco_2017_unlabeled/rgbd_new_label'
    #GROUNDTRUTH_PATH = 'Image_data/sky_groundtruth'
    # Quickshift relies on a C long. As this is platform dependent, I will change
    # this later.
    SEED = 134
    POPULATION = 1000
    GENERATIONS = 100
    MUTATION = 0
    FLIPPROB = 0
    CROSSOVER = 0

    VALIDATION_PATH="" # Not sure how this is used    
    def __init__(self, argv=[]):
        if argv:
            self.parseinput(argv)
    
    """Function to parse the command line inputs"""
    def parseinput(self, argv):   
        # The input arguments are Seed, population, generations, mutation, flipprob, crossover
        print("Parsing Inputs")

        self.SEED = int(argv[1])
        self.MUTATION = float(argv[3])
        self.FLIPPROB = float()

        try:
            self.SEED = int(argv[1])
        except ValueError:
            print("Incorrect SEED value, please input an integer")
        try:
            self.POPULATION = int(argv[2])
            assert self.POPULATION > 0
        except ValueError:
            print("Incorrect POPULATION value: Please input a positive integer.")
            sys.exit(2)
        except AssertionError:
            print("Incorrect POPULATION value: Please input a positve integer.")
            sys.exit(2)

        try:
            self.GENERATIONS = int(argv[3])
        except ValueError:
            print("Incorrect value for GENERATIONS. Please input a positive integer.")
            sys.exit(2)
        except AssertionError:
            print("Incorrect value for GENERATIONS. Please input a positive integer.")
            sys.exit(2)

        try:
            self.MUTATION = float(argv[4])
            assert 0 <= self.MUTATION <= 1

        except ValueError:
            print("Please make sure that MUTATION is a positive percentage (decimal).")
            sys.exit(2)
        except AssertionError:
            print("Please make sure that MUTATION is a positive percentage (decimal).")
            sys.exit(2)

        try:
            self.FLIPPROB = float(argv[5])
            assert 0 <= self.FLIPPROB <= 1
        except ValueError:
            print("Incorrect value for FLIPPROB. Please input a positive percentage (decimal).")
            sys.exit(2)
        except AssertionError:
            print("Incorrect value for FLIPPROB. Please input a positive percentage (decimal).")
            sys.exit(2)

        try:
            self.CROSSOVER = float(argv[6])
            assert 0 <= self.CROSSOVER <= 1
        except ValueError:
            print(
                "Incorrect value for CROSSOVER. Please input a positive percentage (decimal).")
            sys.exit(2)
        except AssertionError:
            print(
                "Incorrect value for CROSSOVER. Please input a positive percentage (decimal).")
            sys.exit(2)


        # Checking the directories
        if (FileClass.check_dir(self.IMAGE_PATH) == False):
            print('ERROR: Directory \"%s\" does not exist' % self.IMAGE_PATH)
            sys.exit(1)

        if(FileClass.check_dir(self.GROUNDTRUTH_PATH) == False):
            print("ERROR: Directory \"%s\" does not exist" % self.VALIDATION_PATH)
            sys.exit(1)

        return 

    # TODO rewrite "main" as part of a class or function structure.
    # TODO rewrite to make it pleasently parallel.
    """Function to run the main GA search function"""
    def runsearch(self):

        # Need to error check these

        initTime = time.time()
        #TODO: Seeting random seed to maxside seems wrong. Why would you do this?
        # To determine the seed for debugging purposes
        seed = self.SEED 
        #seed = random.randrange(sys.maxsize)
        random.seed(seed)
        print("Seed was:", seed)

        # Will later have user input to find where the images are

        # TODO: Take these out and change to getting one image

        # Making an ImageData object for all of the regular images
        AllImages = [ImageData.ImageData(os.path.join(root, name)) for
                     root, dirs, files in os.walk(self.IMAGE_PATH) for name in files]

        # Making an ImageData object for all of the labeled images
        GroundImages = [ImageData.ImageData(os.path.join(root, name)) for
                        root, dirs, files in os.walk(self.GROUNDTRUTH_PATH) for name in
                        files]

        # image_number = 0
        # AllImages = [ImageData.ImageData(os.path.join(root, files[image_number])) for
        # 	root, dirs, files in os.walk(IMAGE_PATH)]
        # GroundImages = [ImageData.ImageData(os.path.join(root, files[image_number])) for
        # 	root, dirs, files in os.walk(GROUNDTRUTH_PATH)]

        # Let's get all possible values in lists

        # TODO: Make seed point input parameter
        # Getting the seedpoint for floodfill
        # Dimensions of the image
        x = AllImages[0].getShape()[0]
        y = AllImages[0].getShape()[1]

        # Multichannel?
        z = 0
        if (AllImages[0].getDim() > 2):
            z = AllImages[0].getShape()[2] - 1

        seedX = [ix for ix in range(0, x)]
        seedY = [iy for iy in range(0, y)]
        seedZ = [z]

        # ADD VALUES FOR NEW PARAMETERS HERE
        #Used in mutate
        AllVals = AlgoHelp().allVals()

        if len(AllVals) == 22:
            # We can just put the seed point at the end
            AllVals.append(seedX)
            AllVals.append(seedY)
            AllVals.append(seedZ)
        else:
            AllVals.insert(22, seedX)
            AllVals.insert(23, seedY)
            AllVals.insert(24, seedZ)

        '''[Algos, betas, tolerance, scale, sigma, min_size,
                  n_segments, compactness, iterations, ratio, kernel, 
                  max_dists, random_seed, connectivity, mu, Lambdas, dt,
                  init_level_set_chan, init_level_set_morph, smoothing,
                  alphas, balloon, seedX, seedY, seedZ]
        '''
        # Using the DEAP genetic algorithm to make One Max
        # https://deap.readthedocs.io/en/master/api/tools.html
        # Creator factory builds new classes

        toolbox = AlgoHelp().makeToolbox(self.POPULATION, seedX, seedY, seedZ)

        # Here we check if we have a saved state
        # From: https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html
        pop = None

        # Keeps track of the best individual from any population
        hof = None
        start_gen = 0

        # TODO: Use copy better
        Images = [AllImages[0] for i in range(0, self.POPULATION)]
        GroundImages = [GroundImages[0] for i in range(0, self.POPULATION)]
        # TODO: Implement a save-state function:
        # https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html

        '''try:
            #A file name was given, so we load it
            with open(sys.argv[1], "r") as cp_file:
                cp = pickle.load(cp_file)
            pop = cp["population"]
            fitnesses = list(map(toolbox.evaluate, Images, GroundImages, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            start_gen = cp["generation"]
            hof = cp["halloffame"]
            random.setstate(cp["rndstate"])
        except IndexError:
            pop = toolbox.population()
            fitnesses = list(map(toolbox.evaluate, Images, GroundImages, pop))

            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            hof = deap.tools.HallOfFame(1)
        '''

        pop = toolbox.population()
        # takes a lot of time
        fitnesses = list(map(toolbox.evaluate, Images, GroundImages, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof = deap.tools.HallOfFame(1)

        #Algo = AlgorithmSpace(AlgoParams)
        extractFits = [ind.fitness.values[0] for ind in pop]
        hof.update(pop)
        # print('OLD: ', extractFits)
        #stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        #stats.register("avg", np.mean)

        # cxpb = probability of two individuals mating
        # mutpb = probability of mutation
        # ngen = Number of generations

        cxpb, mutpb, ngen = self.CROSSOVER, self.MUTATION, self.GENERATIONS
        gen = 0

        leng = len(pop)
        mean = sum(extractFits) / leng
        sum1 = sum(i*i for i in extractFits)
        stdev = abs(sum1 / leng - mean ** 2) ** 0.5
        print(" Min: ", min(extractFits))
        print(" Max: ", max(extractFits))
        print(" Avg: ", mean)
        print(" Std: ", stdev)
        print(" Size: ", leng)
        print(" Time: ", time.time() - initTime)

        # Beginning evolution
        pastPop = pop
        pastMean = mean
        pastMin = min(extractFits)

        BestAvgs = []

        # while min(extractFits) > 0 and gen < ngen:
        # TODO: Think about changing algorithm to:
        # Calc fitness
        # Update population
        while gen < ngen:
            gen += 1
            print("Generation: ", gen)
            offspring = toolbox.select(pop, len(pop))
            #offspring = toolbox.select(pop, 2)
            offspring = list(map(toolbox.clone, offspring))  # original code
            # offspring = [toolbox.clone(ind) for ind in offspring] # changed to

            # crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # Do we crossover?
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    # The parents may be okay values so we should keep them
                    # in the set
                    del child1.fitness.values
                    del child2.fitness.values

            # mutation
            for mutant in offspring:
                if random.random() < mutpb:
                    flipProb = self.FLIPPROB
                    toolbox.mutate(mutant, AllVals, flipProb)
                    del mutant.fitness.values

            # Let's just evaluate the mutated and crossover individuals
            # if not ind.fitness.valid] # was "if not ind.fitness.valid"
            invalInd = [ind for ind in offspring]
            NewImage = [AllImages[0] for i in range(0, len(invalInd))]
            NewVal = [GroundImages[0] for i in range(0, len(invalInd))]
            fitnesses = map(toolbox.evaluate, NewImage, NewVal, invalInd)
            # fitnesses = list(map(toolbox.evaluate, NewImage, NewVal, invalInd))#mycode
            # fitnesses = toolbox.map(toolbox.evaluate, NewImage, NewVal, invalInd) #mycode

            for ind, fit in zip(invalInd, fitnesses):
                ind.fitness.values = fit

            # Replacing the old population
            pop[:] = offspring

            hof.update(pop)
            extractFits = [ind.fitness.values[0] for ind in pop]
            # print(hof[0])
            # print(extractFits)
            # hof.update(pop)

            # Evaluating the new population
            leng = len(pop)
            mean = sum(extractFits) / leng
            BestAvgs.append(mean)
            sum1 = sum(i*i for i in extractFits)
            stdev = abs(sum1 / leng - mean ** 2) ** 0.5
            print(" Min: ", min(extractFits))
            print(" Max: ", max(extractFits))
            print(" Avg: ", mean)
            print(" Std: ", stdev)
            print(" Size: ", leng)
            print(" Time: ", time.time() - initTime)
            print("Best Fitness: ", hof[0].fitness.values)
            print(hof[0])
            # Did we improve the population?
            pastPop = pop
            pastMin = min(extractFits)
            if (mean >= pastMean):
                # This population is worse than the one we had before

                if hof[0].fitness.values[0] <= 0.0001:
                    # The best fitness function is pretty good
                    break
                else:
                    continue
            pastMean = mean

            # TODO: use tools.Statistics for this stuff

        # We ran the population 'ngen' times. Let's see how we did:
        # Now let's checkpoint
        cp = dict(population=pop, generation=gen,
                  halloffame=hof, rndstate=random.getstate())
        best = hof[0]

        with open("checkpoint_name.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)
        print("Best Fitness: ", hof[0].fitness.values)
        print(hof[0])

        finalTime = time.time()
        diffTime = finalTime - initTime
        print("Final time: %.5f seconds" % diffTime)

        # And let's run the algorithm to get an image
        Space = AlgorithmSpace(AlgorithmParams.AlgorithmParams(AllImages[0], best))
        img = Space.runAlgo()
        cv2.imwrite("dummy.png", img)

        # Let's put the best algos into a file. Can later graph with matplotlib.
        file = open("newfile.txt", "a+")
        for i in BestAvgs:
            file.write(str(i) + "\n")
        file.close()
    
if __name__ == '__main__':
    ga = SegmentImage(sys.argv)
    ga.runsearch()
    
