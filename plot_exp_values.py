import numpy as np 
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns

from utilities import *
# plotting endtime M protein value 
times = np.linspace(0,18)
mean_theta = np.log(0.002) # barely an increase: 10 to 14 over the course of 6 months
theta = np.log(0.04) # already unrealistically big: 10 to 20 over 18 days
gr = np.exp(theta)
Mprot = 10*np.exp(gr*times)
fig, ax = plt.subplots()
ax.plot(times, Mprot)
#ax.plot(times, np.log(0.04)*sigmoid(times))
#ax.plot(times, times/(np.sqrt(1+times**2)))
plt.show()
