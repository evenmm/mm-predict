import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from datetime import datetime

filename = './MM_all_patients.xls'
df = pd.read_excel(filename)
print(len(df))
print(df.loc[0,['nnid']][0])
print(df.loc[len(df)-1,['nnid']][0])
print(df.head(n=5))
#for col in df.columns:
#    print(col)

# Columns with dates, drugs or mproteins
df_mprotein_and_dates = df[['nnid', 
'Diagnosis date', 'Serum mprotein (SPEP)', 
'Treatment start', 'Serum mprotein (SPEP) (g/l):', 
'Date of best response:', 'Serum mprotein:', 
'Date of best response:.1', 'Serum mprotein:.1', 
'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Start date', 'End date',
'Date of best respone:', 'Serum mprotein:.2',
'Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1', 'Start date.1', 'End date.1',
'Date of best respone:.1', 'Serum mprotein:.3', 
'Progression date:', 'Serum mprotein:.4',
'DateOfLabValues', 'SerumMprotein',
'Last-data-entered', 'DateOfDeath']]
# Unclear for Serum mprotein: and ... .1
# After Start date End date Date of best respone, the immediate Serum mprotein belongs to Date of best respone.
# 'last follow-up-date' not used because less complete than DateOfLabValues
#print(df_mprotein_and_dates.head(n=5))

# For each patient (nnid), we make a figure and plot all the values
fig, ax1 = plt.subplots()
# Initialize nnid
nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
for row_index in range(len(df_mprotein_and_dates)): # Set 10 to only print nr 1
    # Check if it's the same patient:
    if (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
        # If yes, then plot Mprotein values at corresponding dates
        dates = df_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best response:', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
        mprotein_levels = df_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
        #print(nnid)
        #print(mprotein_levels)
        #print(dates)
        # Suppress missing data for mprotein
        nan_mask_mprotein = np.array(mprotein_levels.notna())
        dates = dates[nan_mask_mprotein]
        mprotein_levels = mprotein_levels[nan_mask_mprotein]
        # and for dates
        nan_mask_dates = np.array(dates.notna())
        dates = dates[nan_mask_dates]
        mprotein_levels = mprotein_levels[nan_mask_dates]

        ax1.plot(dates, mprotein_levels, linestyle='', marker='o')

        # Add drug labeling
        # Use start and end dates of treatment

        #treat_starts = np.array("the sliced panda frame with start times")
        #treat_durations = np.array("end - start") #[20, 200, 200]
        #y_dummy = [0, 1, 2]

        ax2 = ax1.twinx() 
        ax2.set_ylabel("Treatment")
        #plt.barh()
    else:
        # If no, then save plot and initialize new figure with new nnid 
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        fig.tight_layout()
        plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
        #plt.show()
        plt.close()
        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        fig, ax1 = plt.subplots()

ax1.set_title("Patient ID " + str(nnid))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/L)")
fig.tight_layout()
plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
#plt.show()
plt.close()
