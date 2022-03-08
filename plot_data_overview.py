import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from matplotlib.patches import Rectangle
import datetime 
from pandas import DataFrame
import seaborn as sns
from color_dictionaries import *
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

#################################################################################################################
## Make dictionary of drugs, where the key is the drug names, the value is the id
#################################################################################################################
raw_unique_drugs = pd.unique(df_mprotein_and_dates[['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', 'Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']].values.ravel('K'))
nan_mask_drugs = ~isNaN(raw_unique_drugs)
unique_drugs = raw_unique_drugs[nan_mask_drugs]
drug_ids = range(len(unique_drugs))
number_of_unique_drugs = len(unique_drugs)
print("There are "+str(number_of_unique_drugs)+" unique drugs.")
print(unique_drugs)
drug_dictionary = dict(zip(unique_drugs, drug_ids))
print(drug_dictionary)

#################################################################################################################
## Make dictionary of treatment lines, where the key is a frozenset of drug names, the value is the id
#################################################################################################################
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
        ## Exclude Radiation
        #drugs_1 = drugs_1[drugs_1 != "Radiation"]
        # Create an empty list of drugs. Here we assume that there are no duplicate drug entries.
        this_drug_set = []
        for ii in range(len(drugs_1)):
            #drugkey = drug_dictionary[drugs_1[ii]]
            #this_drug_set.append(drugkey)
            this_drug_set.append(drugs_1[ii]) 

    all_treatment_lines.append(frozenset(this_drug_set)) # adding a frozenset to the list of frozensets

unique_treatment_lines = []
for item in all_treatment_lines:
    if item not in unique_treatment_lines:
        unique_treatment_lines.append(item)
# Then create treatment line ids and zip them with the frozenset to create a dict 
treatment_line_ids = range(len(unique_treatment_lines))
number_of_unique_treatment_lines = len(unique_treatment_lines)
print("\nThere are "+str(number_of_unique_treatment_lines)+" unique treatment_lines.")
#print(unique_treatment_lines)
treatment_line_dictionary = dict(zip(unique_treatment_lines, treatment_line_ids))
print(treatment_line_dictionary)

#################################################################################################################
## Make dataframe with treatment lines per patient
#################################################################################################################
# Add a column with treatment line ids
# Not looking at single day treatments of Melphalan; 2-day of Melphalan+Carfilzomib, or Cyclophosphamide: 'Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1'
df_mprotein_and_dates["Treatment line id"] = -1
for row_index in range(len(df_mprotein_and_dates)):
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    missing_date_bool = isNaN(drug_interval_1).any()
    if not missing_date_bool: 
        raw_drugs = df_mprotein_and_dates.loc[row_index, ['Drug 1', 'Drug 2', 'Drug 3', 'Drug 4']].values
        raw_drugs = raw_drugs[~isNaN(raw_drugs)]
        #raw_drugs = raw_drugs[raw_drugs!="Radiation"]
        #print(raw_drugs)
        #print(treatment_line_dictionary[frozenset(raw_drugs)])
        df_mprotein_and_dates.loc[row_index, "Treatment line id"] = treatment_line_dictionary[frozenset(raw_drugs)]

# Sort df after nnid and then Start date
df_mprotein_and_dates = df_mprotein_and_dates.sort_values(['nnid', 'Start date'])
# Reset index to make iteration over sorted version possible
df_mprotein_and_dates.reset_index(drop=True, inplace=True)
print(df_mprotein_and_dates[['nnid', 'Start date', 'End date', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', "Treatment line id"]].head(n=20))

# Now ask: How many patients got Len+Dex+Bor as first treatment? For this we must know 1st, 2nd, 3rd treatment 
# Create new dataframe with 1 row for each patient and columns "Treatment line 1", "Treatment line 2", etc: 
# Since df is sorted by start date of treatment, we can loop over df and take treatment by treatment for each patient, add to this list 
empty_dict = {"nnid": [],
              "Treatment line 1": [],
              "Treatment line 2": [],
              "Treatment line 3": [],
              "Treatment line 4": [],
              "Treatment line 5": [],
              "Treatment line 6": [],
              "Treatment line 7": [],
              "Treatment line 8": [],
              "Treatment line 9": [],
              "Treatment line 10": [],
              "Treatment line 11": [],
              "Treatment line 12": [],
              "Treatment line 13": [],
              "Treatment line 14": [],
              "Treatment line 15": [],
              "Treatment line 16": [],
              "Treatment line 17": [],
              "Treatment line 18": [],
              "Treatment line 19": [],
              "Treatment line 20": []}
df_treatment_lines = pd.DataFrame(empty_dict)

nnid = df_mprotein_and_dates.loc[1,['nnid']][0]
index_counter = 0
next_treatment_line_this_patient = 1
initialized = False
for row_index in range(len(df_mprotein_and_dates)):
    treat_dates = np.array(df_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    missing_date_bool = isNaN(drug_interval_1).any()
    # Check if radiation or missing date
    if (not missing_date_bool) and (not (df_mprotein_and_dates.loc[row_index,['Drug 1']][0] == "Radiation")):
        this_treatment_line_id = df_mprotein_and_dates.loc[row_index,['Treatment line id']][0]
        # If new nnid:
        if not (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
            nnid = df_mprotein_and_dates.loc[row_index,['nnid']][0]
            df_treatment_lines.loc[len(df_treatment_lines.index)] = [nnid, this_treatment_line_id] + np.repeat(np.nan, 19).tolist()
            next_treatment_line_this_patient = 2
            index_counter = index_counter + 1
        else:
            if initialized == False:
                # Add first row to dataframe
                df_treatment_lines.loc[len(df_treatment_lines.index)] = [nnid, this_treatment_line_id] + np.repeat(np.nan, 19).tolist()
                next_treatment_line_this_patient = 2
                initialized = True
            else: 
                # Add treatment line to existing patient
                df_treatment_lines.loc[index_counter, "Treatment line "+str(next_treatment_line_this_patient)] = this_treatment_line_id
                next_treatment_line_this_patient = next_treatment_line_this_patient + 1
print(df_treatment_lines[["nnid", "Treatment line 1", "Treatment line 2", "Treatment line 3", "Treatment line 4", "Treatment line 5", "Treatment line 6", "Treatment line 7", "Treatment line 8", "Treatment line 9"]].head(n=200))
df_treatment_lines.reset_index(drop=True, inplace=True)

# Histogram of Treatment line 1
#treatment_counts = data_treat_line_1.value_counts()
#print(treatment_counts)
data_treat_line_1 = df_treatment_lines[["Treatment line 1"]]
sns.set(font_scale=0.6)
data_treat_line_1.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Treatment line id", labelpad=14)
plt.xticks()
plt.ylabel("Count of patients", labelpad=14)
plt.title("Count of patients per treatment line id for first treatment", y=1.02)
for i, v in enumerate(data_treat_line_1.value_counts()):
    plt.text(i*1.004-0.16, v+1, str(v))
plt.savefig("./first_treatment_line_id.png")
#plt.show()
plt.close()

#plt.bar(range(len(treatment_counts)), treatment_counts)
#plt.show()

# Histogram of Treatment line 2 for those that got 'Velcade (bortezomib) - subcut twice weekly', 'Dexamethasone', 'Cyclophosphamide' as first treatment line
df_VelDexCyclo_as_first_treatment = df_treatment_lines.loc[df_treatment_lines['Treatment line 1'] == 2] 
print("These receive treatment line 2 as first treatment")
pd.set_option('display.max_rows', 70)
print(df_VelDexCyclo_as_first_treatment) #[["nnid", "Treatment line 1", "Treatment line 2", "Treatment line 3", "Treatment line 4", "Treatment line 5", "Treatment line 6", "Treatment line 7", "Treatment line 8", "Treatment line 9"]].head(n=200))

plt.figure()
data_treat_line_2 = df_VelDexCyclo_as_first_treatment[["Treatment line 2"]]
sns.set(font_scale=0.6)
data_treat_line_2.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)
plt.xlabel("Treatment line id", labelpad=14)
plt.xticks()
plt.ylabel("Count of patients", labelpad=14)
plt.title("Count of patients per treatment line id for second treatment", y=1.02)
for i, v in enumerate(data_treat_line_2.value_counts()):
    plt.text(i*1.004-0.16, v+1, str(v))
plt.savefig("./second_treatment_line_given_VelDexCyclo.png")
#plt.show()
plt.close()

# Plot the M protein vales for those patients that receive treatment 2 first, then treatment 1
# Choose only the lines from the sorted first dataframe that match these nnid
selected_nnid = df_VelDexCyclo_as_first_treatment["nnid"].tolist()
print("selected_nnid", selected_nnid)
df_selected_mprotein_and_dates = df_mprotein_and_dates.loc[df_mprotein_and_dates['nnid'].isin(selected_nnid)]
df_selected_mprotein_and_dates.reset_index(drop=True, inplace=True)
print("These are the patients that receive treatment line 2 as first treatment:")
print(df_selected_mprotein_and_dates[['nnid', 'Start date', 'End date', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', "Treatment line id"]].head(n=20))
print("These are their second treatments")
print(df_VelDexCyclo_as_first_treatment[['Treatment line 2']].values.ravel('K'))

treatment_starts = []

# Find the treatment start we are interested in: The start of second treatment
correct_patient_history = [2,1]
nnid = df_selected_mprotein_and_dates.loc[1,['nnid']][0]
patient_history = []
patient_count = 0
for row_index in range(len(df_selected_mprotein_and_dates)):
    treat_dates = np.array(df_selected_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    missing_date_bool = isNaN(drug_interval_1).any()
    # Check if radiation or missing date
    if (not missing_date_bool) and (not (df_selected_mprotein_and_dates.loc[row_index,['Drug 1']][0] == "Radiation")):
        this_treatment_line_id = df_selected_mprotein_and_dates.loc[row_index,['Treatment line id']][0]
        # Check if it's the same patient.
        # If it's a new patient, then start checking fistory from beginning
        if not (df_selected_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
            nnid = df_selected_mprotein_and_dates.loc[row_index,['nnid']][0]
            patient_history = []
        patient_history.append(this_treatment_line_id)

        # Patient history defines the history + the one we want to plot
        # If the history matches the correct one then the last is the one we want to plot. Save its treatment start
        if patient_history == correct_patient_history:
        #if len(patient_history) == len(correct_patient_history)+1:
            # Save the treatment start date
            treatment_starts.append(np.array(df_selected_mprotein_and_dates.loc[row_index, ['Treatment start']])[0])
            patient_count = patient_count + 1 
print(treatment_starts)
print(patient_count)
# Give each patient a color: 
patient_colordict = dict(zip(selected_nnid, treat_line_colors[0:len(selected_nnid)]))

# Then plot all the M protein values with time relative to the treatment start
# Iterate through lines of treatment with M protein values
# For each patient, loop over the filtered df_mprotein_and_dates:
#   Plot the M protein values from the treatment line we are interested in (e.g. first or second time they have Dex+Len+Velcade)
print(df_selected_mprotein_and_dates[['nnid', 'Diagnosis date', 'Serum mprotein (SPEP)', 'Treatment start', 'Serum mprotein (SPEP) (g/l):', 'Date of best response:', 'Serum mprotein:', 'Date of best response:.1', 'Serum mprotein:.1', 'Date of best respone:', 'Serum mprotein:.2', 'Date of best respone:.1', 'Serum mprotein:.3', 'Progression date:', 'Serum mprotein:.4', 'DateOfLabValues', 'SerumMprotein']].head(n=20))
nnid = df_selected_mprotein_and_dates.loc[1,['nnid']][0]
patient_history = []
fig, ax1 = plt.subplots()
#plt.setp(ax1.xaxis.get_minorticklabels(), rotation=90)
#plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
ax1.patch.set_facecolor('none')
ax1.axvline(0, color="k", linewidth=0.5, linestyle="-")
#ax1.xaxis.set_major_locator(years)
#ax1.xaxis.set_major_formatter(yearsFmt)
#ax1.xaxis.set_minor_locator(months)
#ax1.xaxis.set_minor_formatter(monthsFmt)
ax2 = ax1.twinx() 
plotheight = 1
patient_count = 0
for row_index in range(len(df_selected_mprotein_and_dates)):
    treat_dates = np.array(df_selected_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    missing_date_bool = isNaN(drug_interval_1).any()
    # Check if radiation or missing date
    if (not missing_date_bool) and (not (df_selected_mprotein_and_dates.loc[row_index,['Drug 1']][0] == "Radiation")):
        this_treatment_line_id = df_selected_mprotein_and_dates.loc[row_index,['Treatment line id']][0]
        # Check if it's the same patient.
        # If it's a new patient, then start checking fistory from beginning
        if not (df_selected_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
            nnid = df_selected_mprotein_and_dates.loc[row_index,['nnid']][0]
            patient_history = []
        patient_history.append(this_treatment_line_id)

        if patient_history == correct_patient_history:
        #if len(patient_history) == len(correct_patient_history):
            # It's the last treatment and we plot the M protein values
            # All:
            #dates = df_selected_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best response:', 'Date of best response:.1', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
            #mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
            # Or just some: 
            dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best response:', 'Date of best response:.1', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
            mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]

            # Suppress cases with missing data for mprotein
            nan_mask_mprotein = np.array(mprotein_levels.notna())
            dates = dates[nan_mask_mprotein]
            mprotein_levels = mprotein_levels[nan_mask_mprotein]
            # and for dates
            nan_mask_dates = np.array(dates.notna())
            dates = dates[nan_mask_dates]
            mprotein_levels = mprotein_levels[nan_mask_dates]

            # Reset time to time after Treatment start
            if len(dates) > 0:
                adjusted_dates = (dates - np.repeat(treatment_starts[patient_count], len(dates))).dt.days
                #adjusted_dates = dates - np.repeat(treatment_starts[patient_count], len(dates))

                #adjusted_dates = adjusted_dates.to_datetime()
                if len(adjusted_dates) > 0:
                    if max(abs(adjusted_dates)) > 3000:
                    #if max(abs(adjusted_dates)) > datetime.timedelta(days=3000):
                        print(type(adjusted_dates))
                        print(adjusted_dates)

                ax1.plot(adjusted_dates, mprotein_levels, linestyle='-', linewidth=1, marker='x', markersize=0.5, markeredgecolor="k", zorder=3, color=patient_colordict[nnid])
            patient_count = patient_count + 1 
ax1.set_title("Patients receiving combo " + str(correct_patient_history[0]) + " after combo " + str(correct_patient_history[1]))
ax1.set_xlabel("Time (days since treatment start)")
ax1.set_ylabel("Serum Mprotein (g/L)")
ax1.set_ylim(bottom=0)
#ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
ax1.set_zorder(ax1.get_zorder()+3)
#fig.autofmt_xdate()
fig.tight_layout()
plt.savefig("./history_" + str(correct_patient_history) + ".png")
plt.show()
plt.close()


################################################################################################################
# Plot tree of treatment line histories 
#################################################################################################################

"""
#################################################################################################################
# Individual drugs and M protein history plot
#################################################################################################################
# For each patient (nnid), we make a figure and plot all the values
drug_colordict = dict(zip(drug_ids, drug_colors))
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
            ax2.add_patch(Rectangle((drug_interval_1[0], drugkey - plotheight/2), treatment_time_1, plotheight, zorder=2, color=drug_colordict[drugkey]))
            #else:
            #ax2.plot(drug_interval_1, [drugkey, drugkey], linestyle='-', linewidth=10, marker='D', zorder=2, color=drug_colordict[drugkey])

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
            ax2.add_patch(Rectangle((drug_interval_2[0], drugkey - plotheight/2), treatment_time_2, plotheight, zorder=2, color=drug_colordict[drugkey]))
            #else:
            #ax2.plot(drug_interval_2, [drugkey, drugkey], linestyle='-', linewidth=10, marker='D', zorder=2, color=drug_colordict[drugkey])

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
# For each patient (nnid), we make a figure and plot the treatment lines
treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
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
max_treat_line_key = 0
patient_count = 0
for row_index in range(len(df_mprotein_and_dates)):
    # Check if it's the same patient.
    # If it's a new patient, then save plot and initialize new figure with new nnid
    if not (df_mprotein_and_dates.loc[row_index,['nnid']][0] == nnid):
        ax1.set_title("Patient ID " + str(nnid))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/L)")
        ax1.set_ylim(bottom=0)
        ax2.set_ylabel("Treatment line")
        ax2.set_yticks(range(max_treat_line_key+1))
        ax2.set_yticklabels(range(max_treat_line_key+1))
        #ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
        ax1.set_zorder(ax1.get_zorder()+3)
        fig.autofmt_xdate()
        fig.tight_layout()
        # Only save the plot if there are at least 3 M protein measurements
        if count_mprotein > 2 and count_treatments > 0:
            patient_count = patient_count + 1
            plt.savefig("./Treatment_line_plots/" + str(nnid) + ".png")
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
        max_treat_line_key = 0

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
        this_drug_set = []
        for ii in range(len(drugs_1)):
            #drugkey = drug_dictionary[drugs_1[ii]]
            #this_drug_set.append(drugkey)
            this_drug_set.append(drugs_1[ii])
        # Find the treatment line id
        this_treatment_line = frozenset(this_drug_set)
        treat_line_id = treatment_line_dictionary[this_treatment_line]
        if treat_line_id > max_treat_line_key:
            max_treat_line_key = treat_line_id
        ax2.add_patch(Rectangle((drug_interval_1[0], treat_line_id - plotheight/2), treatment_time_1, plotheight, zorder=2, color=treat_colordict[treat_line_id]))

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
ax2.set_ylabel("Treatment line")
ax2.set_yticks(range(max_treat_line_key+1))
ax2.set_yticklabels(range(max_treat_line_key+1))
#ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
ax1.set_zorder(ax1.get_zorder()+3)
fig.autofmt_xdate()
fig.tight_layout()
if count_mprotein > 2 and count_treatments > 0:
    patient_count = patient_count + 1
    plt.savefig("./Treatment_line_plots/" + str(nnid) + ".png")
#plt.show()
plt.close()

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
"""
