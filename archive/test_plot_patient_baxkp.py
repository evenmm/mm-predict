import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from datetime import datetime
from pandas import DataFrame
import seaborn as sns

fig, ax1 = plt.subplots()

#dts = s.index.to_pydatetime()   ## AttributeError: 'numpy.ndarray' object has no attribute 'to_pydatetime'
#ax.plot(dts, s)
ax1.patch.set_facecolor('none')
ax2 = ax1.twinx() 

ax1.plot([0,1,2,3,4], [1,2,3,4,5], linestyle='', marker='x', zorder=3, color='k')

ax2.plot([1,3], [2, 2], linestyle='-', linewidth=27.8, marker='D', zorder=2, color="red")

ax1.set_title("Patient ID " + str(1))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/L)")
ax1.set_ylim(bottom=0)
ax2.set_ylabel("Drug")
#ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
ax1.set_zorder(ax1.get_zorder()+3)
fig.tight_layout()
plt.savefig("./testplot_1_3.png")
plt.show()
plt.close()
