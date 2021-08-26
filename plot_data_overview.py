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
nan_mask_treatments = ~isNaN(unique_treatments)
unique_treatments = unique_treatments[nan_mask_treatments]
drugkeys = range(len(unique_treatments))
print("There are "+str(len(unique_treatments))+" unique treatments.")
#print(unique_treatments)
treatment_dictionary = dict(zip(unique_treatments, drugkeys))
print(treatment_dictionary)

# Make a color dictionary
# Unique colors: http://godsnotwheregodsnot.blogspot.com/2012/09/color-distribution-methodology.html
colors = np.array([
[0, 0, 0],
[1, 0, 103],
[213, 255, 0],
[255, 0, 86],
[158, 0, 142],
[14, 76, 161],
[255, 229, 2],
[0, 95, 57],
[0, 255, 0],
[149, 0, 58],
[255, 147, 126],
[164, 36, 0],
[0, 21, 68],
[145, 208, 203],
[98, 14, 0],
[107, 104, 130],
[0, 0, 255],
[0, 125, 181],
[106, 130, 108],
[0, 174, 126],
[194, 140, 159],
[190, 153, 112],
[0, 143, 156],
[95, 173, 78],
[255, 0, 0],
[255, 0, 246],
[255, 2, 157],
[104, 61, 59],
[255, 116, 163],
[150, 138, 232]
])/255
colordict = dict(zip(drugkeys, colors))

# For each patient (nnid), we make a figure and plot all the values
fig, ax1 = plt.subplots()
ax1.patch.set_facecolor('none')
ax2 = ax1.twinx() 
ymaxvalue = 0
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
        # Remove cases with missing end dates
        missing_date_bool = isNaN(drug_interval_1).any()
        if not missing_date_bool: 
            # Remove nan drugs
            drugs_1 = drugs_1[~isNaN(drugs_1)]
            drug1_codes = np.zeros(len(drugs_1))
            for ii in range(len(drugs_1)):
                drug1_codes[ii] = treatment_dictionary[drugs_1[ii]]
            for ii in range(len(drugs_1)):
                drugkey = treatment_dictionary[drugs_1[ii]]
                #print(drugkey)
                ax2.plot(drug_interval_1, [drugkey, drugkey], linestyle='-', linewidth=10, marker='d', zorder=2, color=colordict[drugkey])

        drug_interval_2 = treat_dates[2:4] # For the second drug combination
        drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])
        # Remove cases with missing end dates
        missing_date_bool = isNaN(drug_interval_2).any()
        if not missing_date_bool: 
            # Remove nan drugs
            drugs_2 = drugs_2[~isNaN(drugs_2)]
            drug2_codes = np.zeros(len(drugs_2))
            for ii in range(len(drugs_2)):
                drug2_codes[ii] = treatment_dictionary[drugs_2[ii]]
            # Remove drug_codes after debug
            
            #print(drug1_codes)
            #print(drug2_codes)
            #print(drug_interval_1)
            #print(drug_interval_2)

            for ii in range(len(drugs_2)):
                drugkey = treatment_dictionary[drugs_2[ii]]
                ax2.plot(drug_interval_2, [drugkey, drugkey], linestyle='-', linewidth=10, marker='d', zorder=2, color=colordict[drugkey])

        # Plot Mprotein values at corresponding dates
        dates = df_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best response:', 'Date of best response:.1', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
        mprotein_levels = df_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
        #print(nnid)
        # Suppress missing data for mprotein
        nan_mask_mprotein = np.array(mprotein_levels.notna())
        dates = dates[nan_mask_mprotein]
        mprotein_levels = mprotein_levels[nan_mask_mprotein]
        # and for dates
        nan_mask_dates = np.array(dates.notna())
        dates = dates[nan_mask_dates]
        mprotein_levels = mprotein_levels[nan_mask_dates]
        #print(mprotein_levels)
        #print(dates)

        #if max(mprotein_levels) > ymaxvalue:
        #    ymaxvalue = max(mprotein_levels)

        ax1.plot(dates, mprotein_levels, linestyle='', marker='x', zorder=3) #, color='k')

    else:
        # If no, then save plot and initialize new figure with new nnid 
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        #ax1.set_ylim([0, 1.05*ymaxvalue])
        ax2.set_ylabel("Treatment")
        #ax2.set_ylim([-0.5,len(unique_treatments)+0.5]) # If you want to cover all unique treatments
        ax1.set_zorder(ax1.get_zorder()+3)
        fig.tight_layout()
        plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
        #plt.show()
        plt.close()

        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        fig, ax1 = plt.subplots()
        ax1.patch.set_facecolor('none')
        ax2 = ax1.twinx() 
        ymaxvalue = 0

ax1.set_title("Patient ID " + str(nnid))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/L)")
#ax1.set_ylim([0, 1.05*ymaxvalue])
ax2.set_ylabel("Treatment")
#ax2.set_ylim([-0.5,len(unique_treatments)+0.5]) # If you want to cover all unique treatments
ax1.set_zorder(ax1.get_zorder()+3)
fig.tight_layout()
plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
#plt.show()
plt.close()
