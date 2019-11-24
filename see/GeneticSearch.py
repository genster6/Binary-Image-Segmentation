import random

from deap import algorithms
from deap import base
from deap import tools
from deap import creator
from scoop import futures

from see import Segmentors


# Executes a crossover between two numpy arrays of the same length
def twoPointCopy(np1, np2):
    assert(len(np1) == len(np2))
    size = len(np1)
    point1 = random.randint(1, size)
    point2 = random.randint(1, size-1)
    if (point2 >= point1):
        point2 += 1
    else:  # Swap the two points
        point1, point2 = point2, point1
    np1[point1:point2], np2[point1:point2] = np2[point1:point2].copy(), np1[point1:point2].copy()
    return np1, np2

'''Executes a crossover between two arrays (np1 and np2) picking a
random amount of indexes to change between the two.
'''
def skimageCrossRandom(np1, np2):
    # TODO: Only change values associated with algorithm
    assert(len(np1) == len(np2))
    # The number of places that we'll cross
    crosses = random.randrange(len(np1))
    # We pick that many crossing points
    indexes = random.sample(range(0, len(np1)), crosses)
    # And at those crossing points, we switch the parameters

    for i in indexes:
        np1[i], np2[i] = np2[i], np1[i]

    return np1, np2

''' Changes a few of the parameters of the weighting a random
    number against the flipProb.
    Variables:
    copyChild is the individual to mutate
    posVals is a list of lists where each list are the possible
        values for that particular parameter
    flipProb is how likely, it is that we will mutate each value.
        It is computed seperately for each value.
'''
def mutate(copyChild, posVals, flipProb=0.5):

    # Just because we chose to mutate a value doesn't mean we mutate
    # Every aspect of the value
    child = copy.deepcopy(copyChild)

    # Not every algorithm is associated with every value
    # Let's first see if we change the algorithm
    randVal = random.random()
    if randVal < flipProb:
        # Let's mutate
        child[0] = random.choice(posVals[0])
    # Now let's get the indexes (parameters) related to that value
    switcher = AlgoHelp().algoIndexes()
    indexes = switcher.get(child[0])

    for index in indexes:
        randVal = random.random()
        if randVal < flipProb:
            # Then we mutate said value
            if index == 22:
                # Do some special
                X = random.choice(posVals[22])
                Y = random.choice(posVals[23])
                Z = random.choice(posVals[24])
                child[index] = (X, Y, Z)
                continue

            child[index] = random.choice(posVals[index])
    return child


def makeToolbox(pop_size):

    seedX = 10;
    seedY = 10;
    seedZ = 20;
    
    #Minimizing fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-0.000001,))

    creator.create("Individual", list, fitness=creator.FitnessMin)

    #The functions that the GA knows
    toolbox = base.Toolbox()
    
#     #Attribute generator
#     toolbox.register("attr_bool", random.randint, 0, 1000)

    #Genetic functions
    toolbox.register("mate", skimageCrossRandom) #crossover
    toolbox.register("mutate", mutate) #Mutation
    toolbox.register("evaluate", Segmentors.runAlgo) #Fitness

    toolbox.register("select", tools.selTournament, tournsize=5) #Selection
    # toolbox.register("select", tools.selBest)
    toolbox.register("map", futures.map) #So that we can use scoop
    
    #TODO: May want to later do a different selection process

    #Here we register all the parameters to the toolbox
#     SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT = 0, 1, 0.5
    
    #Perhaps weight iterations
#     ITER = 10
#     SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT = 1, 4, 0.5
#     BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT = -1, 1, 0.9

    #We choose the parameters, for the most part, random
    params = Segmentors.parameters()
    
    for key in params.pkeys:
        toolbox.register(key, random.choice, eval(params.ranges[key]))
    
#     #smoothing should be 1-4, but can be any positive number
#     toolbox.register("attr_smooth", RandHelp.weighted_choice,
#         self.PosVals[19], SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT)
#     toolbox.register("attr_alphas", random.choice, self.PosVals[20])
#     #Should be from -1 to 1, but can be any value
#     toolbox.register("attr_balloon", RandHelp.weighted_choice, 
#         self.PosVals[21], BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT)

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
    
    func_seq = []
#     for key in params.pkeys:
#         print(f"toolbox.{key}")
        
#     func_seq = [ eval(f"toolbox.{key}") for key in params.pkeys ]
#     func_seq = [toolbox.attr_Algo, toolbox.attr_Beta, toolbox.attr_Tol,
#         toolbox.attr_Scale, toolbox.attr_Sigma, toolbox.attr_minSize,
#         toolbox.attr_nSegment, 
#         toolbox.attr_iterations, toolbox.attr_ratio,
#         toolbox.attr_kernel, toolbox.attr_maxDist, toolbox.attr_seed, 
#         toolbox.attr_connect, toolbox.attr_compact, toolbox.attr_mu, 
#         toolbox.attr_lambda, toolbox.attr_dt, toolbox.attr_init_chan,
#         toolbox.attr_init_morph, toolbox.attr_smooth, 
#         toolbox.attr_alphas, toolbox.attr_balloon, 
#         toolbox.attr_seed_pointX, toolbox.attr_seed_pointY,
#         toolbox.attr_seed_pointZ]

    #AT THE END OF THE 'func_seq' ADD MORE PARAMETERS
    #print(func_seq)
    #Here we populate our individual with all of the parameters
    toolbox.register("individual", tools.initCycle, creator.Individual, func_seq, n=1)


    #And we make our population
    toolbox.register("population", tools.initRepeat, list, 
        toolbox.individual, n=pop_size)


    #def makeToolBox(self):
    return toolbox


