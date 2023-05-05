import numpy as np 
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import seaborn as sns

SAVEDIR = "./plots/Bayesian_estimates_simdata_BNN/"

fig, ax1 = plt.subplots()
shade_array = [0.7, 0.5, 0.35]
for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
    ax1.fill_between([-1,0,1], [1,-1,-1], [1,-1,-1], color=plt.cm.copper(1-critical_value), label='%3.0f %% CI, resistant M protein' % (100*(1-2*critical_value)), zorder=0+index*0.1)
for index, critical_value in enumerate([0.05, 0.25, 0.45]): # Corresponding to confidence levels 90, 50, and 10
    ax1.fill_between([-1,0,1], [1,-1,-1], [1,-1,-1], color=plt.cm.bone(shade_array[index]), label='%3.0f %% CI, total M protein' % (100*(1-2*critical_value)), zorder=1+index*0.1)
ax1.plot([-1,0,1], [1,-1,-1], linestyle='--', marker='', zorder=3, color='cyan', label="True M protein (total)")
ax1.plot([-1,0,1], [1,-1,-1], linestyle='--', marker='', zorder=2.9, color=plt.cm.hot(0.2), label="True M protein (resistant)")
ax1.plot([-1,0,1], [1,-1,-1], linestyle='', marker='x', zorder=4, color='k', label="Observed M protein") #[ax1.axvline(time, color="k", linewidth=0.5, linestyle="-") for time in measurement_times]
plt.legend(loc="upper right")
plt.savefig(SAVEDIR+"xxxxAUC_.pdf", dpi=300)
plt.show()