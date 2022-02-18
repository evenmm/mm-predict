import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from datetime import datetime
from pandas import DataFrame
import seaborn as sns
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')

filename = './MM_all_patients.xls'
print("Loading data frame from file:", filename)
df = pd.read_excel(filename)
print("Number of rows in dataframe:", len(df))
print("Position of patient ID column 'nnid':", df.loc[0,['nnid']][0])
print("nnid of last patient:", df.loc[len(df)-1,['nnid']][0])
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
raw_unique_treatments = pd.unique(df_mprotein_and_dates[['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']].values.ravel('K'))
nan_mask_treatments = ~isNaN(raw_unique_treatments)
unique_treatments = raw_unique_treatments[nan_mask_treatments]
drugkeys = range(len(unique_treatments))
number_of_unique_drugs = len(unique_treatments)
print("There are "+str(number_of_unique_drugs)+" unique treatments.")
print(unique_treatments)
treatment_dictionary = dict(zip(unique_treatments, drugkeys))
print(treatment_dictionary)

# Make a color dictionary
# Unique colors: http://godsnotwheregodsnot.blogspot.com/2012/09/color-distribution-methodology.html
# 0 1 12 14
#[0, 0, 0],
#[1, 0, 103],
#[0, 21, 68],
#[98, 14, 0],
colors = np.array([
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
[145, 208, 203],
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
[150, 138, 232],
[152, 255, 82],
[167, 87, 64],
[1, 255, 254],
[255, 238, 232],
[254, 137, 0],
[189, 198, 255],
[1, 208, 255],
[187, 136, 0],
[117, 68, 177],
[165, 255, 210],
[255, 166, 254],
[119, 77, 0],
[122, 71, 130],
[38, 52, 0],
[0, 71, 84],
[67, 0, 44],
[181, 0, 255],
[255, 177, 103],
[255, 219, 102],
[144, 251, 146],
[126, 45, 210],
[189, 211, 147],
[229, 111, 254],
[222, 255, 116],
[0, 255, 120],
[0, 155, 255],
[0, 100, 1],
[0, 118, 255],
[133, 169, 0],
[0, 185, 23],
[120, 130, 49],
[0, 255, 198],
[255, 110, 65],
[232, 94, 190]
])/255
colordict = dict(zip(drugkeys, colors))

# For each patient (nnid), we make a figure and plot all the values
# Initialize nnid
nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
count_mprotein = 0
count_treatments = 0
fig, ax1 = plt.subplots()
ax1.patch.set_facecolor('none')
ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(yearsFmt)
ax1.xaxis.set_minor_locator(months)
ax2 = ax1.twinx() 
maxdrugkey = 0
for row_index in range(len(df_mprotein_and_dates)):
    # Check if it's the same patient.
    # If it's a new patient, then save plot and initialize new figure with new nnid
    if not (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        ax1.set_ylim(bottom=0)
        ax2.set_ylabel("Treatment")
        ax2.set_yticks(range(maxdrugkey+1))
        ax2.set_yticklabels(range(maxdrugkey+1))
        #ax2.set_ylim([-0.5,len(unique_treatments)+0.5]) # If you want to cover all unique treatments
        ax1.set_zorder(ax1.get_zorder()+3)
        fig.autofmt_xdate()
        fig.tight_layout()
        # Only save the plot if there are 3 consecutive M protein measurements under treatment
        if count_mprotein > 2 and count_treatments > 0:
            plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
        #plt.show()
        plt.close()

        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        count_mprotein = 0
        count_treatments = 0
        fig, ax1 = plt.subplots()
        ax1.patch.set_facecolor('none')
        ax1.xaxis.set_major_locator(years)
        ax1.xaxis.set_major_formatter(yearsFmt)
        ax1.xaxis.set_minor_locator(months)
        ax2 = ax1.twinx() 
        maxdrugkey = 0

    # Plot treatments
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])

    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    drugs_1 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_1).any()
    if not missing_date_bool: 
        count_treatments = count_treatments+1
        # Remove nan drugs
        drugs_1 = drugs_1[~isNaN(drugs_1)]
        for ii in range(len(drugs_1)):
            drugkey = treatment_dictionary[drugs_1[ii]]
            if drugkey > maxdrugkey:
                maxdrugkey = drugkey
            ax2.plot(drug_interval_1, [drugkey, drugkey], linestyle='-', linewidth=27.8, marker='D', zorder=2, color=colordict[drugkey])

    drug_interval_2 = treat_dates[2:4] # For the second drug combination
    drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_2).any()
    if not missing_date_bool: 
        count_treatments = count_treatments+1
        # Remove nan drugs
        drugs_2 = drugs_2[~isNaN(drugs_2)]
        for ii in range(len(drugs_2)):
            drugkey = treatment_dictionary[drugs_2[ii]]
            if drugkey > maxdrugkey:
                maxdrugkey = drugkey
            ax2.plot(drug_interval_2, [drugkey, drugkey], linestyle='-', linewidth=27.8, marker='D', zorder=2, color=colordict[drugkey])

    # Plot Mprotein values at corresponding dates
    dates = df_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best response:', 'Date of best response:.1', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
    mprotein_levels = df_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
    # Suppress cases with missing data for mprotein
    nan_mask_mprotein = np.array(mprotein_levels.notna())
    dates = dates[nan_mask_mprotein]
    mprotein_levels = mprotein_levels[nan_mask_mprotein]
    # and for dates
    nan_mask_dates = np.array(dates.notna())
    dates = dates[nan_mask_dates]
    mprotein_levels = mprotein_levels[nan_mask_dates]
    if len(mprotein_levels) > 0:
        count_mprotein = count_mprotein+1

    ax1.plot(dates, mprotein_levels, linestyle='', marker='x', zorder=3, color='k')

ax1.set_title("Patient ID " + str(nnid))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/L)")
ax1.set_ylim(bottom=0)
ax2.set_ylabel("Treatment")
ax2.set_yticks(range(maxdrugkey+1))
ax2.set_yticklabels(range(maxdrugkey+1))
#ax2.set_ylim([-0.5,len(unique_treatments)+0.5]) # If you want to cover all unique treatments
ax1.set_zorder(ax1.get_zorder()+3)
fig.autofmt_xdate()
fig.tight_layout()
if count_mprotein > 0 and count_treatments > 0:
    plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
#plt.show()
plt.close()

# Plot matrix of drug counts per patient
# dictionary of drugs: treatment_dictionary = dict(zip(unique_treatments, drugkeys))
unique_nnids = pd.unique(df_mprotein_and_dates[['nnid']].values.ravel('K'))
print(unique_nnids)
nnid_dict_keys = range(len(unique_nnids))
nnid_dictionary = dict(zip(unique_nnids, nnid_dict_keys))
number_of_nnids = len(unique_nnids)
print("There are "+str(number_of_nnids)+" unique nnids.")
#%matplotlib inline
Index= unique_nnids
Cols = unique_treatments
drugmatrix = DataFrame(np.zeros((number_of_nnids,number_of_unique_drugs)), index=Index, columns=Cols)

# Loop over rows. For each line, find the drug names, lookup in dict and update matrix 
# Initialize nnid
nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
nnid_key = nnid_dictionary[nnid] # The dataframe row index
for row_index in range(len(df_mprotein_and_dates)):
    # Check if it's the same patient.
    if not (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        nnid_key = nnid_dictionary[nnid] # The dataframe row index

    # Find dates of treatment
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])

    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    drugs_1 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_1).any()
    if not missing_date_bool: 
        # Remove nan drugs
        drugs_1 = drugs_1[~isNaN(drugs_1)]
        for ii in range(len(drugs_1)):
            drugkey = treatment_dictionary[drugs_1[ii]]
            #df.at['C', 'x'] = 10
            drugmatrix.iloc[nnid_key,drugkey] = drugmatrix.iloc[nnid_key,drugkey] + 1 

    drug_interval_2 = treat_dates[2:4] # For the second drug combination
    drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_2).any()
    if not missing_date_bool: 
        # Remove nan drugs
        drugs_2 = drugs_2[~isNaN(drugs_2)]
        for ii in range(len(drugs_2)):
            drugkey = treatment_dictionary[drugs_2[ii]]
            drugmatrix.iloc[nnid_key,drugkey] = drugmatrix.iloc[nnid_key,drugkey] + 1 

# Count matrix showing how many times drug was given to patient
plt.figure(figsize=(15,15))
sns.heatmap(drugmatrix.iloc[:,:], annot=False)
plt.tight_layout()
plt.savefig("./drug_matrix.png")
plt.show()

# Binary matrix indicating whether the drug was used
plt.figure(figsize=(15,15))
sns.heatmap(drugmatrix.iloc[:,:]>0, annot=False)
plt.tight_layout()
plt.savefig("./drug_matrix_binary.png")
plt.show()
