import argparse
import sys

parser = argparse.ArgumentParser(description='Run the see-Semgenta algorithm')
parser.add_argument("-g", "--generations", help="Number of Generations to run in search", type=int, default=10)
parser.add_argument("-s", "--seed", help="Random Seed", type=int, default=0)
parser.add_argument("--mutation", help="Mutation Rate (0 - 100)", type=float, default=0.5)
parser.add_argument("--flipprob", help="Flip Probability", type=float, default=0.5)
parser.add_argument("--crossover", help="Crossover Rate", type=float, default=0.5)

parser.add_argument("-p", "--pop", help="Population (file or number)", type=str, default="100")
parser.add_argument("-i", "--image", help="Input image file", type=str, default="")
parser.add_argument("-m", "--mask", help="Mask ground truth", type=str, default="")

args = parser.parse_args()
print(args)

