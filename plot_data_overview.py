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
# Remember to update dates and mprotein levels below

# Unclear for Serum mprotein: and ... .1
# After Start date End date Date of best respone, the immediate Serum mprotein belongs to Date of best respone.
# 'last follow-up-date' not used because less complete than DateOfLabValues
#print(df_mprotein_and_dates.head(n=5))

def isNaN(string):
    return string != string

# Make dictionary of different treatments. This will be a color code
unique_treatments = pd.unique(df_mprotein_and_dates[['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']].values.ravel('K'))
unique_dummies = range(len(unique_treatments))
#print(unique_treatments)
treatment_dictionary = dict(zip(unique_treatments, unique_dummies))

# Make a color dictionary

# For each patient (nnid), we make a figure and plot all the values
fig, ax1 = plt.subplots()
# Initialize nnid
nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
for row_index in range(len(df_mprotein_and_dates)):
    # Check if it's the same patient:
    if (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
        # If yes:
        # Plot treatments
        treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])

        drug_interval_1 = treat_dates[0:2] # For the first drug combination
        drugs_1 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']])
        # Remove nan
        drugs_1 = drugs_1[~isNaN(drugs_1)]
        drug1_codes = np.zeros(len(drugs_1))
        for ii in range(len(drugs_1)):
            drug1_codes[ii] = treatment_dictionary[drugs_1[ii]]

        drug_interval_2 = treat_dates[2:4] # For the second drug combination
        drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])
        # Remove nan
        drugs_2 = drugs_2[~isNaN(drugs_2)]
        drug2_codes = np.zeros(len(drugs_2))
        for ii in range(len(drugs_2)):
            drug2_codes[ii] = treatment_dictionary[drugs_2[ii]]
        
        print(drug1_codes)
        print(drug2_codes)
        print(drug_interval_1)
        print(drug_interval_2)

        ax2 = ax1.twinx() 
        ax2.set_ylabel("Treatment")
        for ii in range(len(drug1_codes)):
            ax2.plot(drug_interval_1, [drug1_codes[ii], drug1_codes[ii]], linestyle='-', linewidth=10, marker='d', zorder=1)
        for ii in range(len(drug2_codes)):
            ax2.plot(drug_interval_2, [drug2_codes[ii], drug2_codes[ii]], linestyle='-', linewidth=10, marker='d', zorder=1)
        #ax2.grid(which='both', axis='y')
        ax2.set_ylim([-0.5,len(unique_treatments)+0.5])

        # Plot Mprotein values at corresponding dates
        dates = df_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best response:', 'Date of best response:.1', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
        mprotein_levels = df_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
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

        ax1.plot(dates, mprotein_levels, linestyle='', marker='x', zorder=2) #, color='k')
    else:
        # If no, then save plot and initialize new figure with new nnid 
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        fig.tight_layout()
        plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
        plt.show()
        plt.close()
        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        fig, ax1 = plt.subplots()

ax1.set_title("Patient ID " + str(nnid))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/L)")
fig.tight_layout()
plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
plt.show()
plt.close()
