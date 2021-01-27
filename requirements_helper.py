import sys
sys.stdout = open('requirements.txt', 'w')


import numpy
print(f"numpy=={numpy.__version__}")
import torch
print(f"torch=={torch.__version__}")
import gym
print(f"gym=={gym.__version__}")
import matplotlib
print(f"matplotlib=={matplotlib.__version__}")
import sympy
print(f"sympy=={sympy.__version__}")
import torchdiffeq
print(f"torchdiffeq=={torchdiffeq.__version__}")

