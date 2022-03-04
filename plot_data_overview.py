import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from matplotlib.patches import Rectangle
import datetime 
from pandas import DataFrame
import seaborn as sns
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')
#monthsFmt = mdates.DateFormatter('%b')
#yearsFmt = mdates.DateFormatter('\n\n%Y')  # add some space for the year label

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

## Make dictionary of drugs
raw_unique_drugs = pd.unique(df_mprotein_and_dates[['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']].values.ravel('K'))
nan_mask_drugs = ~isNaN(raw_unique_drugs)
unique_drugs = raw_unique_drugs[nan_mask_drugs]
drug_ids = range(len(unique_drugs))
number_of_unique_drugs = len(unique_drugs)
print("There are "+str(number_of_unique_drugs)+" unique drugs.")
print(unique_drugs)
drug_dictionary = dict(zip(unique_drugs, drug_ids))
print(drug_dictionary)

## Make dictionary of treatment lines
# Make a list of treatment lines. A treatment line is a frozenset of drug_ids, making this a list of frozensets
all_treatment_lines = []
# Loop over all treatment lines and add them to the set
for row_index in range(len(df_mprotein_and_dates)):
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])

    # Treatment over a period:
    drugs_1 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']])
    # One-day treatments:
    drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])

    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    treatment_time_1 = drug_interval_1[1] - drug_interval_1[0]
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_1).any()
    if not missing_date_bool: 
        # Remove nan drugs
        drugs_1 = drugs_1[~isNaN(drugs_1)]
        # Create an empty list of drugs. Here we assume that there are no duplicate drug entries.
        this_drug_set = []
        for ii in range(len(drugs_1)):
            drugkey = drug_dictionary[drugs_1[ii]]
            this_drug_set.append(drugkey)
            # Could also add drug name: drugs_1[ii]

    all_treatment_lines.append(frozenset(this_drug_set)) # adding a frozenset to the list of frozensets

# Then create a frozenset from the list so entries are unique and hashable
unique_treatment_lines = frozenset(all_treatment_lines)
# Then create treatment line ids and zip them with the frozenset to create a dict 
treatment_line_ids = range(len(unique_treatment_lines))
number_of_unique_treatment_lines = len(unique_treatment_lines)
print("There are "+str(number_of_unique_treatment_lines)+" unique treatment_lines.")
print(unique_treatment_lines)
treatment_line_dictionary = dict(zip(unique_treatment_lines, treatment_line_ids))
print(treatment_line_dictionary)

#################################################################################################################
# Individual drugs and M protein history plot
#################################################################################################################
# For each patient (nnid), we make a figure and plot all the values
from color_dictionaries import *
colordict = dict(zip(drug_ids, drug_colors))
# Initialize nnid
nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
count_mprotein = 0
count_treatments = 0
fig, ax1 = plt.subplots()
#plt.setp(ax1.xaxis.get_minorticklabels(), rotation=90)
#plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
ax1.patch.set_facecolor('none')
ax1.xaxis.set_major_locator(years)
ax1.xaxis.set_major_formatter(yearsFmt)
ax1.xaxis.set_minor_locator(months)
#ax1.xaxis.set_minor_formatter(monthsFmt)
ax2 = ax1.twinx() 
plotheight = 1
maxdrugkey = 0
patient_count = 0
for row_index in range(len(df_mprotein_and_dates)):
    # Check if it's the same patient.
    # If it's a new patient, then save plot and initialize new figure with new nnid
    if not (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        ax1.set_ylim(bottom=0)
        ax2.set_ylabel("Drug")
        ax2.set_yticks(range(maxdrugkey+1))
        ax2.set_yticklabels(range(maxdrugkey+1))
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
        ax1.set_zorder(ax1.get_zorder()+3)
        fig.autofmt_xdate()
        fig.tight_layout()
        # Only save the plot if there are at least 3 M protein measurements
        if count_mprotein > 2 and count_treatments > 0:
            patient_count = patient_count + 1
            plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
        #plt.show()
        plt.close()

        nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
        count_mprotein = 0
        count_treatments = 0
        fig, ax1 = plt.subplots()
        #plt.setp(ax1.xaxis.get_minorticklabels(), rotation=90)
        #plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        ax1.patch.set_facecolor('none')
        ax1.xaxis.set_major_locator(years)
        ax1.xaxis.set_major_formatter(yearsFmt)
        ax1.xaxis.set_minor_locator(months)
        #ax1.xaxis.set_minor_formatter(monthsFmt)
        ax2 = ax1.twinx() 
        maxdrugkey = 0

    # Plot treatments
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])

    # First drug entry: Treatments that last more than 1 day 
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    treatment_time_1 = drug_interval_1[1] - drug_interval_1[0]
    #print(treatment_time_1)
    drugs_1 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_1).any()
    if not missing_date_bool: 
        count_treatments = count_treatments+1
        # Remove nan drugs
        drugs_1 = drugs_1[~isNaN(drugs_1)]
        for ii in range(len(drugs_1)):
            drugkey = drug_dictionary[drugs_1[ii]]
            if drugkey > maxdrugkey:
                maxdrugkey = drugkey
            #if treatment_time_1 > datetime.timedelta(days=10):
            #             Rectangle(             xy              , width           , height, angle=0.0, ...)
            ax2.add_patch(Rectangle((drug_interval_1[0], drugkey - plotheight/2), treatment_time_1, plotheight, zorder=2, color=colordict[drugkey]))
            #else:
            #ax2.plot(drug_interval_1, [drugkey, drugkey], linestyle='-', linewidth=10, marker='D', zorder=2, color=colordict[drugkey])

    # Second drug entry: Single day treatments
    drug_interval_2 = treat_dates[2:4] # For the second drug combination
    treatment_time_2 = drug_interval_2[1] - drug_interval_2[0]
    #print(treatment_time_2)
    drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_2).any()
    if not missing_date_bool: 
        count_treatments = count_treatments+1
        # Remove nan drugs
        drugs_2 = drugs_2[~isNaN(drugs_2)]
        for ii in range(len(drugs_2)):
            drugkey = drug_dictionary[drugs_2[ii]]
            if drugkey > maxdrugkey:
                maxdrugkey = drugkey
            #if treatment_time_1 > datetime.timedelta(days=10):
            #             Rectangle(             xy              , width           , height, angle=0.0, ...)
            ax2.add_patch(Rectangle((drug_interval_2[0], drugkey - plotheight/2), treatment_time_2, plotheight, zorder=2, color=colordict[drugkey]))
            #else:
            #ax2.plot(drug_interval_2, [drugkey, drugkey], linestyle='-', linewidth=10, marker='D', zorder=2, color=colordict[drugkey])

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
ax2.set_ylabel("Drug")
ax2.set_yticks(range(maxdrugkey+1))
ax2.set_yticklabels(range(maxdrugkey+1))
#ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
ax1.set_zorder(ax1.get_zorder()+3)
fig.autofmt_xdate()
fig.tight_layout()
if count_mprotein > 2 and count_treatments > 0:
    patient_count = patient_count + 1
    plt.savefig("./Mproteinplots/" + str(nnid) + ".png")
#plt.show()
plt.close()

print("There are "+str(patient_count)+" patients that satisfy the criteria for inclusion.")

#################################################################################################################
# Treatment lines and M protein history plot
#################################################################################################################

#---


#################################################################################################################
# Plot matrix of drug counts per patient
#################################################################################################################
# dictionary of drugs: drug_dictionary = dict(zip(unique_drugs, drug_ids))
unique_nnids = pd.unique(df_mprotein_and_dates[['nnid']].values.ravel('K'))
print(unique_nnids)
nnid_dict_keys = range(len(unique_nnids))
nnid_dictionary = dict(zip(unique_nnids, nnid_dict_keys))
number_of_nnids = len(unique_nnids)
print("There are "+str(number_of_nnids)+" unique nnids.")
#%matplotlib inline
Index= unique_nnids
Cols = unique_drugs
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
            drugkey = drug_dictionary[drugs_1[ii]]
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
            drugkey = drug_dictionary[drugs_2[ii]]
            drugmatrix.iloc[nnid_key,drugkey] = drugmatrix.iloc[nnid_key,drugkey] + 1 

# Count matrix showing how many times drug was given to patient
plt.figure(figsize=(15,15))
sns.heatmap(drugmatrix.iloc[:,:], annot=False)
plt.tight_layout()
plt.savefig("./drug_matrix.png")
#plt.show()

# Binary matrix indicating whether the drug was used
plt.figure(figsize=(15,15))
sns.heatmap(drugmatrix.iloc[:,:]>0, annot=False)
plt.tight_layout()
plt.savefig("./drug_matrix_binary.png")
#plt.show()
