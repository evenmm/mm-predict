import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from datetime import datetime

filename = './MM_all_patients.xls'
password = b'iSo2lgniMb' # Have to encode it as byte string

df = pd.read_excel(filename)


print(df.head(n=5))

# Columns with either dates or mproteins
#df_mprotein_and_dates = df[['nnid', 'Diagnosis date', 'Serum mprotein (SPEP)', 'Treatment start', 'Serum mprotein (SPEP) (g/l)', 'Serum mprotein:', 'Date of best response:', 'Serum mprotein:.1', 'Date of best respone:', 'Serum mprotein:.2', 'Date of best respone:.1', 'Serum mprotein:.3', 'Progression date:', 'Serum mprotein:.4']]

#df.rename(columns={'Serum mprotein (SPEP)':'Serum mprotein Diagnosis date', 
#'Serum mprotein (SPEP) (g/l)':'Serum mprotein Treatment start', 
#'Serum mprotein:.1':'Serum mprotein Date of best response', 
#'Serum mprotein:.2':'Serum mprotein Date of best respone:', 
#'Serum mprotein:.3':'Serum mprotein Date of best respone:.1', 
#'Serum mprotein:.4':'Serum mprotein Progression date',
#'SerumMprotein':'Serum mprotein DateOfLabValues'})

#df_mprotein_and_dates = df[['nnid', 'Diagnosis date', 'Serum mprotein Diagnosis date', 
#'Treatment start', 'Serum mprotein Treatment start', 
#'Date of best response:', 'Serum mprotein Date of best response', 
#'Date of best respone:', 'Serum mprotein Date of best respone:', 
#'Date of best respone:.1', 'Serum mprotein Date of best respone:.1', 
#'Progression date:', 'Serum mprotein Progression date',
#'DateOfLabValues', 'Serum mprotein DateOfLabValues']]

df_mprotein_and_dates = df[['nnid', 'Diagnosis date', 'Serum mprotein (SPEP)', 
'Treatment start', 'Serum mprotein (SPEP) (g/l):', 
'Date of best response:', 'Serum mprotein:.1', 
'Date of best respone:', 'Serum mprotein:.2', 
'Date of best respone:.1', 'Serum mprotein:.3', 
'Progression date:', 'Serum mprotein:.4',
'DateOfLabValues', 'SerumMprotein']]

#df.rename(columns={
#'Serum mprotein (SPEP)':'Serum mprotein Diagnosis date', 
#'Serum mprotein (SPEP) (g/l):':'Serum mprotein Treatment start', 
#'Serum mprotein:.1':'Serum mprotein Date of best response', 
#'Serum mprotein:.2':'Serum mprotein Date of best respone:', 
#'Serum mprotein:.3':'Serum mprotein Date of best respone:.1', 
#'Serum mprotein:.4':'Serum mprotein Progression date',
#'SerumMprotein':'Serum mprotein DateOfLabValues'})

#for col in df.columns:
#    print(col)

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
        #print(dates)
        mprotein_levels = df_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
        #print(mprotein_levels)

        # Ignore missing data
        nan_mask = np.array(mprotein_levels.notna())
        #print(nan_mask)
        #print(mprotein_levels[nan_mask])
        #print(dates[nan_mask])
        ax1.plot(dates[nan_mask], mprotein_levels[nan_mask], linestyle='', marker='o')
        #plt.plot(dates[nan_mask], mprotein_levels[nan_mask], linestyle='', marker='o')

        # Add drug labeling
        # Use start and end dates of treatment

        #treat_starts = np.array("the sliced panda frame with start times")
        #treat_durations = np.array("end - start") #[20, 200, 200]
        #y_dummy = [0, 1, 2]

        ax2 = ax1.twinx() 
        ax2.set_ylabel("Treatment")
        #plt.barh()
    else:
        # Save plot and initialize new figure with new nnid 
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
        plt.show()
        plt.close()
        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        fig, ax1 = plt.subplots()

ax1.set_title("Patient ID " + str(nnid))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/L)")
plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
plt.show()
plt.close()
