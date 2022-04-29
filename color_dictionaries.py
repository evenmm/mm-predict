from drug_colors import *
from treat_line_colors import *
import numpy as np
import matplotlib.pyplot as plt
# Make a color dictionary
# Unique colors: http://godsnotwheregodsnot.blogspot.com/2012/09/color-distribution-methodology.html
# 0 1 12 14
#[0, 0, 0],
#[1, 0, 103],
#[0, 21, 68],
#[98, 14, 0],
# Two lists in separate files
print(len(drug_colors))

extension_floats = np.linspace(0,1,num=24)

extension = np.array([plt.cm.plasma(i)[0:3] for i in extension_floats])
#print(treat_line_colors)
#print(extension)
treat_line_colors = np.concatenate((treat_line_colors, extension))
#print(treat_line_colors)
#print(len(treat_line_colors))
