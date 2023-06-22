import numpy as np 
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns

from utilities import *
# plotting endtime M protein value 
times = np.linspace(0,400)
theta_rho_r = np.log(0.001) # already unrealistically big: 10 to 20 over 18 days
gr = np.exp(theta_rho_r)
Mprot = 10*np.exp(gr*times)
fig, ax = plt.subplots()
ax.plot(times, Mprot, "r")
theta_rho_s = np.log(0.05) # already unrealistically big: 10 to 20 over 18 days
gs = np.exp(theta_rho_s)
sMprot = 50*np.exp(-gs*times)
ax.plot(times, sMprot, "b")
#ax.plot(times, np.log(0.04)*sigmoid(times))
#ax.plot(times, times/(np.sqrt(1+times**2)))
plt.show()
