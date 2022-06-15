# Importing
from scipy import *
import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import sys
import time

n_lines = 15
evaluationpoints = np.linspace(0, 10)
phase_shift = np.linspace(0, np.pi, n_lines)

color_idx = np.linspace(0, 1, n_lines)
for i, shift in zip(color_idx, phase_shift):
    plt.plot(evaluationpoints, np.sin(evaluationpoints - shift), color=plt.cm.plasma(i), lw=3)
    print(plt.cm.plasma(i))
plt.title("viridis")
plt.show()