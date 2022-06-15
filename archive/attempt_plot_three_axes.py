import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates 
from matplotlib.patches import Rectangle
import datetime 
from pandas import DataFrame
import seaborn as sns
from color_dictionaries import *
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')
#monthsFmt = mdates.DateFormatter('%b')
#yearsFmt = mdates.DateFormatter('\n\n%Y')  # add some space for the year label
def isNaN(string):
    return string != string

# M protein data
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_VISIT_V2.tsv'
print("Loading data frame from file:", filename)
df = pd.read_csv(filename, sep='\t')
print("Number of rows in dataframe:", len(df))
print("PUBLIC_ID of first patient:", df.loc[0,['PUBLIC_ID']][0])
print("PUBLIC_ID of last patient:", df.loc[len(df)-1,['PUBLIC_ID']][0])
print(df.head(n=5))
#for col in df.columns:
#    print(col)
# Columns with dates, drugs or mproteins
df_mprotein_and_dates = df[['PUBLIC_ID', 'VISIT', 'VISITDY',
'D_LAB_serum_m_protein',
'D_IM_LIGHT_CHAIN_BY_FLOW', #"kappa" or "lambda"
'D_LAB_serum_kappa', # Serum Kappa (mg/dL)
'D_LAB_serum_lambda', # Serum Lambda (mg/dL)
'D_IM_kaplam'
]]
print(df_mprotein_and_dates.head(n=5))

# Remove lines with nan times 
df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['VISITDY'].notna()]

raw_unique_nnid = pd.unique(df_mprotein_and_dates[['PUBLIC_ID']].values.ravel('K'))
nan_mask_nnid = ~isNaN(raw_unique_nnid)
unique_nnid = raw_unique_nnid[nan_mask_nnid]
number_of_unique_nnid = len(unique_nnid)
print("There are "+str(number_of_unique_nnid)+" unique patients.")


# Drug data
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_STAND_ALONE_TREATMENT_REGIMEN_V2.tsv'
print("Loading data frame from file:", filename)
df = pd.read_csv(filename, sep='\t')
print("Number of rows in dataframe:", len(df))
print(df.head(n=5))
df_drugs_and_dates = df[[
    'PUBLIC_ID', 'MMTX_THERAPY', 
    'startday', 'stopday'
]]
#################################################################################################################
## Make dictionary of drugs, where the key is the drug names, the value is the id
#################################################################################################################
raw_unique_drugs = pd.unique(df_drugs_and_dates[['MMTX_THERAPY']].values.ravel('K'))
unique_drugs = raw_unique_drugs[~isNaN(raw_unique_drugs)]
drug_ids = range(len(unique_drugs))
number_of_unique_drugs = len(unique_drugs)
print("There are "+str(number_of_unique_drugs)+" unique drugs.")
drug_dictionary = dict(zip(unique_drugs, drug_ids))
print(drug_dictionary)
np.save("drug_dictionary_COMMPASS.npy", drug_dictionary)
#print(df_drugs_and_dates.head(n=5))

"""
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
treatment_to_id_dictionary = dict(zip(unique_treatment_lines, treatment_line_ids))
print(treatment_to_id_dictionary)
np.save("treatment_to_id_dictionary_COMMPASS.npy", treatment_to_id_dictionary)

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
        #print(treatment_to_id_dictionary[frozenset(raw_drugs)])
        df_mprotein_and_dates.loc[row_index, "Treatment line id"] = treatment_to_id_dictionary[frozenset(raw_drugs)]

# Sort df after PUBLIC_ID and then Start date
df_mprotein_and_dates = df_mprotein_and_dates.sort_values(['PUBLIC_ID', 'Start date'])
# Reset index to make iteration over sorted version possible
df_mprotein_and_dates.reset_index(drop=True, inplace=True)
print(df_mprotein_and_dates[['PUBLIC_ID', 'Start date', 'End date', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', "Treatment line id"]].head(n=20))

#################################################################################################################
## Make dataframe with treatment lines per patient
#################################################################################################################
# How many patients got Len+Dex+Bor as first treatment? For this we must know 1st, 2nd, 3rd treatment 
# Create new dataframe with 1 row for each patient and columns "Treatment line 1", "Treatment line 2", etc: 
# Since df is sorted by start date of treatment, we can loop over df and take treatment by treatment for each patient, add to this list 
empty_dict = {"PUBLIC_ID": [],
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

PUBLIC_ID = df_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
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
        # If new PUBLIC_ID:
        if not (df_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
            PUBLIC_ID = df_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]
            df_treatment_lines.loc[len(df_treatment_lines.index)] = [PUBLIC_ID, this_treatment_line_id] + np.repeat(np.nan, 19).tolist()
            next_treatment_line_this_patient = 2
            index_counter = index_counter + 1
        else:
            if initialized == False:
                # Add first row to dataframe
                df_treatment_lines.loc[len(df_treatment_lines.index)] = [PUBLIC_ID, this_treatment_line_id] + np.repeat(np.nan, 19).tolist()
                next_treatment_line_this_patient = 2
                initialized = True
            else: 
                # Add treatment line to existing patient
                df_treatment_lines.loc[index_counter, "Treatment line "+str(next_treatment_line_this_patient)] = this_treatment_line_id
                next_treatment_line_this_patient = next_treatment_line_this_patient + 1
print(df_treatment_lines[["PUBLIC_ID", "Treatment line 1", "Treatment line 2", "Treatment line 3", "Treatment line 4", "Treatment line 5", "Treatment line 6", "Treatment line 7", "Treatment line 8", "Treatment line 9"]].head(n=200))
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
plt.savefig("./COMMPASS_first_treatment_line_id.pdf")
#plt.show()
plt.close()

#plt.bar(range(len(treatment_counts)), treatment_counts)
#plt.show()

# Histogram of Treatment line 2 for those that got 'Velcade (bortezomib) - subcut twice weekly', 'Dexamethasone', 'Cyclophosphamide' as first treatment line
df_VelDexCyclo_as_first_treatment = df_treatment_lines.loc[df_treatment_lines['Treatment line 1'] == 2] 
print("These receive treatment line 2 as first treatment")
pd.set_option('display.max_rows', 70)
print(df_VelDexCyclo_as_first_treatment) #[["PUBLIC_ID", "Treatment line 1", "Treatment line 2", "Treatment line 3", "Treatment line 4", "Treatment line 5", "Treatment line 6", "Treatment line 7", "Treatment line 8", "Treatment line 9"]].head(n=200))

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
plt.savefig("./COMMPASS_second_treatment_line_given_VelDexCyclo.pdf")
#plt.show()
plt.close()

#################################################################################################################
# Plot M protein vales for patients that receive treatment 2 (at any time)
#################################################################################################################
# Choose only the lines from the sorted first dataframe that match these PUBLIC_ID
chosen_treatment_line = 2
print("Choose patients that receive treatment "+str(chosen_treatment_line)+" at any time")
df_selected_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['Treatment line id'] == chosen_treatment_line]
df_selected_mprotein_and_dates.reset_index(drop=True, inplace=True)
selected_PUBLIC_ID = df_selected_mprotein_and_dates["PUBLIC_ID"].unique()
print("These are the patients that receive treatment line 2:")
print(df_selected_mprotein_and_dates[['PUBLIC_ID', 'Start date', 'End date', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', "Treatment line id"]].head(n=20))

# Give each patient a color: 
patient_colordict = dict(zip(selected_PUBLIC_ID, treat_line_colors[0:len(selected_PUBLIC_ID)]))

# Then plot all the M protein values with time relative to the treatment start
# Iterate through lines of treatment with M protein values
# For each patient, loop over the filtered df_mprotein_and_dates:
#   Plot the M protein values from the treatment line we are interested in (e.g. first or second time they have Dex+Len+Velcade)
print(df_selected_mprotein_and_dates[['PUBLIC_ID', 'Diagnosis date', 'Serum mprotein (SPEP)', 'Treatment start', 'Serum mprotein (SPEP) (g/l):', 'Date of best response:', 'Serum mprotein:', 'Date of best response:.1', 'Serum mprotein:.1', 'Date of best respone:', 'Serum mprotein:.2', 'Date of best respone:.1', 'Serum mprotein:.3', 'Progression date:', 'Serum mprotein:.4', 'DateOfLabValues', 'SerumMprotein']].head(n=20))
PUBLIC_ID = df_selected_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
fig, ax1 = plt.subplots()
ax1.patch.set_facecolor('none')
ax1.axvline(0, color="k", linewidth=0.5, linestyle="-")
ax2 = ax1.twinx() 
plotheight = 1
for row_index in range(len(df_selected_mprotein_and_dates)):
    treat_dates = np.array(df_selected_mprotein_and_dates.loc[row_index, ['Start date', 'End date', 'Start date.1', 'End date.1']])
    drug_interval_1 = treat_dates[0:2] # For the first drug combination
    missing_date_bool = isNaN(drug_interval_1).any()
    # Check if radiation (Redundant) or missing date
    if (not missing_date_bool) and (not (df_selected_mprotein_and_dates.loc[row_index,['Drug 1']][0] == "Radiation")):
        this_treatment_line_id = df_selected_mprotein_and_dates.loc[row_index,['Treatment line id']][0]
        # Check if it's the same patient.
        if not (df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
            PUBLIC_ID = df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]

        # Plot the M protein values
        # All:
        #dates = df_selected_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best response:', 'Date of best response:.1', 'Date of best respone:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
        #mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (g/l):', 'Serum mprotein:', 'Serum mprotein:.1', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
        # Or just some: 
        dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:']] #, 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
        treatment_start = dates[0]
        mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2']] #, 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]

        # Check that the value at treatment start exists: 
        if not np.isnan(mprotein_levels[0]):
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
                adjusted_dates = (dates - np.repeat(treatment_start, len(dates))).dt.days
                if len(adjusted_dates) > 0:
                    if max(abs(adjusted_dates)) > 3000:
                        print(type(adjusted_dates))
                        print(adjusted_dates)
                ax1.plot(adjusted_dates, mprotein_levels/np.repeat(mprotein_levels[0], len(mprotein_levels)), linestyle='-', linewidth=1, marker='x', markersize=0.5, markeredgecolor="k", zorder=3, color=patient_colordict[PUBLIC_ID])
ax1.set_title("Patients receiving combo " + str(chosen_treatment_line)+", normalized.")
ax1.set_xlabel("Time (days since treatment start)")
ax1.set_ylabel("Serum Mprotein / Mprotein at treatment start")
ax1.set_ylim(bottom=0)
ax1.set_zorder(ax1.get_zorder()+3)
fig.tight_layout()
plt.savefig("./COMMPASS_response_to_treatment_" + str(chosen_treatment_line) + ".pdf")
#plt.show()
plt.close()




#################################################################################################################
# Plot the M protein vales for those patients that receive treatment 2 first, then treatment 1
#################################################################################################################
# Choose only the lines from the sorted first dataframe that match these PUBLIC_ID
selected_PUBLIC_ID = df_VelDexCyclo_as_first_treatment["PUBLIC_ID"].tolist()
print("selected_PUBLIC_ID", selected_PUBLIC_ID)
df_selected_mprotein_and_dates = df_mprotein_and_dates.loc[df_mprotein_and_dates['PUBLIC_ID'].isin(selected_PUBLIC_ID)]
df_selected_mprotein_and_dates.reset_index(drop=True, inplace=True)
print("These are the patients that receive treatment line 2 as first treatment:")
print(df_selected_mprotein_and_dates[['PUBLIC_ID', 'Start date', 'End date', 'Drug 1', 'Drug 2', 'Drug 3', 'Drug 4', "Treatment line id"]].head(n=20))
print("These are their second treatments")
print(df_VelDexCyclo_as_first_treatment[['Treatment line 2']].values.ravel('K'))

# Find the treatment start we are interested in: The start of second treatment
treatment_starts = []
correct_patient_history = [2]
PUBLIC_ID = df_selected_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
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
        if not (df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
            PUBLIC_ID = df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]
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
# Removed this one to match the one above
#patient_colordict = dict(zip(selected_PUBLIC_ID, treat_line_colors[0:len(selected_PUBLIC_ID)]))

# Then plot all the M protein values with time relative to the treatment start
# Iterate through lines of treatment with M protein values
# For each patient, loop over the filtered df_mprotein_and_dates:
#   Plot the M protein values from the treatment line we are interested in (e.g. first or second time they have Dex+Len+Velcade)
print(df_selected_mprotein_and_dates[['PUBLIC_ID', 'Diagnosis date', 'Serum mprotein (SPEP)', 'Treatment start', 'Serum mprotein (SPEP) (g/l):', 'Date of best response:', 'Serum mprotein:', 'Date of best response:.1', 'Serum mprotein:.1', 'Date of best respone:', 'Serum mprotein:.2', 'Date of best respone:.1', 'Serum mprotein:.3', 'Progression date:', 'Serum mprotein:.4', 'DateOfLabValues', 'SerumMprotein']].head(n=20))
PUBLIC_ID = df_selected_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
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
        if not (df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
            PUBLIC_ID = df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]
            patient_history = []
        patient_history.append(this_treatment_line_id)

        if patient_history == correct_patient_history:
        #if len(patient_history) == len(correct_patient_history):
            # It's the last treatment and we plot the M protein values
            # All:
            #dates = df_selected_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best respone:', 'Date of best response:.1', 'Date of best response:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
            #mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (gSeru.1', 'Serum mm mprotein:', 'Serum mprotein:/l):', 'protein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
            # Or just some: 
            #dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:', 'Date of best response:.1', 'Date of best response:']] #, 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
            #mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2', 'Serum mprotein:.1', 'Serum mprotein:']] #, 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
            dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:', 'Date of best response:.1', 'Date of best response:', 'Date of best respone:.1', 'Progression date:']]#, 'DateOfLabValues']]
            mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2', 'Serum mprotein:.1', 'Serum mprotein:', 'Serum mprotein:.3', 'Serum mprotein:.4']]#, 'SerumMprotein']]
            ##dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:', 'Date of best respone:.1', 'Date of best response:.1']] #, 'Date of best response:']] #, 'Progression date:', 'DateOfLabValues']]
            ##mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.1']] #, 'Serum mprotein:']] #, 'Serum mprotein:.4', 'SerumMprotein']]

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

                # Sort Mprotein values by dates 
                new_dates = [x for x,_ in sorted(zip(adjusted_dates, mprotein_levels))]
                new_mprotein_levels = [y for _,y in sorted(zip(adjusted_dates, mprotein_levels))]
                adjusted_dates = new_dates
                mprotein_levels = new_mprotein_levels

                ax1.plot(adjusted_dates, mprotein_levels/np.repeat(mprotein_levels[0], len(mprotein_levels)), linestyle='-', linewidth=1, marker='x', markersize=0.5, markeredgecolor="k", zorder=3, color=patient_colordict[PUBLIC_ID])
            patient_count = patient_count + 1 

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
        if not (df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
            PUBLIC_ID = df_selected_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]
            patient_history = []
        patient_history.append(this_treatment_line_id)

        if patient_history == correct_patient_history:
        #if len(patient_history) == len(correct_patient_history):
            # It's the last treatment and we plot the M protein values
            # All:
            #dates = df_selected_mprotein_and_dates.loc[row_index, ['Diagnosis date', 'Treatment start', 'Date of best respone:', 'Date of best response:.1', 'Date of best response:', 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
            #mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP)', 'Serum mprotein (SPEP) (gSeru.1', 'Serum mm mprotein:', 'Serum mprotein:/l):', 'protein:.2', 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
            # Or just some: 
            #dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:', 'Date of best response:.1', 'Date of best response:']] #, 'Date of best respone:.1', 'Progression date:', 'DateOfLabValues']]
            #mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2', 'Serum mprotein:.1', 'Serum mprotein:']] #, 'Serum mprotein:.3', 'Serum mprotein:.4', 'SerumMprotein']]
            dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:', 'Date of best response:.1', 'Date of best response:', 'Date of best respone:.1', 'Progression date:']]#, 'DateOfLabValues']]
            mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2', 'Serum mprotein:.1', 'Serum mprotein:', 'Serum mprotein:.3', 'Serum mprotein:.4']]#, 'SerumMprotein']]
            ##dates = df_selected_mprotein_and_dates.loc[row_index, ['Treatment start', 'Date of best respone:', 'Date of best respone:.1', 'Date of best response:.1']] #, 'Date of best response:']] #, 'Progression date:', 'DateOfLabValues']]
            ##mprotein_levels = df_selected_mprotein_and_dates.loc[row_index, ['Serum mprotein (SPEP) (g/l):', 'Serum mprotein:.2', 'Serum mprotein:.3', 'Serum mprotein:.1']] #, 'Serum mprotein:']] #, 'Serum mprotein:.4', 'SerumMprotein']]

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

                # Sort Mprotein values by dates 
                new_dates = [x for x,_ in sorted(zip(adjusted_dates, mprotein_levels))]
                new_mprotein_levels = [y for _,y in sorted(zip(adjusted_dates, mprotein_levels))]
                adjusted_dates = new_dates
                mprotein_levels = new_mprotein_levels

                ax1.plot(adjusted_dates, mprotein_levels/np.repeat(mprotein_levels[0], len(mprotein_levels)), linestyle='', linewidth=1, marker='*', markersize=1, markeredgecolor="k", zorder=3, color=patient_colordict[PUBLIC_ID])
            patient_count = patient_count + 1 
#ax1.set_title("Patients receiving combo " + str(correct_patient_history[0]) + " after combo " + str(correct_patient_history[1]))
ax1.set_title("Patients receiving combo " + str(correct_patient_history[0]) + " as first treatment")
ax1.set_xlabel("Time (days since treatment start)")
#ax1.set_ylabel("Serum Mprotein (g/dL)")
ax1.set_ylabel("Serum Mprotein / Mprotein at treatment start")
ax1.set_ylim(bottom=0)
#ax2.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
ax1.set_zorder(ax1.get_zorder()+3)
#fig.autofmt_xdate()
fig.tight_layout()
plt.savefig("./COMMPASS_history_" + str(correct_patient_history) + ".pdf")
plt.show()
plt.close()
"""
#################################################################################################################
# Individual drugs and M protein history plot
#################################################################################################################
# For each patient (PUBLIC_ID), we make a figure and plot all the values
unique_PUBLIC_IDs = pd.unique(df_mprotein_and_dates[['PUBLIC_ID']].values.ravel('K'))
drug_colordict = dict(zip(drug_dictionary.values(), drug_colors))
plotheight = 1

def plot_patient(PUBLIC_ID):
    # Begin figure
    count_mprotein = 0
    count_treatments = 0
    host = host_subplot(111, axes_class=axisartist.Axes)
    plt.subplots_adjust(left=0.75)
    par1 = host.twinx()
    par1.axis["right"].toggle(all=True)
    par2 = host.twinx()
    par2.axis["left"] = par2.new_fixed_axis(loc="left", offset=(-60, 0))
    par2.axis["left"].toggle(all=True)

    #fig, ax1 = plt.subplots()
    #plt.setp(ax1.xaxis.get_minorticklabels(), rotation=90)
    #plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    host.patch.set_facecolor('none')
    host.xaxis.set_major_locator(years)
    host.xaxis.set_major_formatter(yearsFmt)
    host.xaxis.set_minor_locator(months)
    #host.xaxis.set_minor_formatter(monthsFmt)
    #ax2 = ax1.twinx() 
    maxdrugkey = 0

    # Plot Mprotein values
    for index, row in df_mprotein_and_dates.loc[df_mprotein_and_dates['PUBLIC_ID'] == PUBLIC_ID].iterrows():
        date = row['VISITDY']
        mprotein_value = row['D_LAB_serum_m_protein']
        # Suppress cases with missing data for mprotein
        # nan_mask_mprotein = np.array(mprotein_value.notna())
        # dates = dates[nan_mask_mprotein]
        # mprotein_value = mprotein_value[nan_mask_mprotein]
        # # and for dates
        # nan_mask_dates = np.array(dates.notna())
        # dates = dates[nan_mask_dates]
        # mprotein_value = mprotein_value[nan_mask_dates]
        
        if ~np.isnan(mprotein_value): 
            count_mprotein = count_mprotein+1

        host.plot(date, mprotein_value, linestyle='', marker='x', zorder=3, color='k')
        host.axvline(date, color="k", linewidth=0.5, linestyle="-")

        # Plot light chain (Kappa/Lambda) values at corresponding dates
        kappa_levels = row['D_LAB_serum_kappa']
        par2.plot(date, kappa_levels, linestyle='', marker='x', zorder=2, color='b')
        lambda_levels = row['D_LAB_serum_lambda']
        par2.plot(date, lambda_levels, linestyle='', marker='x', zorder=2, color='r')
    
    # Plot the drugs  ##########################################
    for index, row in df_drugs_and_dates.loc[df_drugs_and_dates['PUBLIC_ID'] == PUBLIC_ID].iterrows():
        # Plot treatments
        treat_dates = np.array([row['startday'], row['stopday']])
    
        # First drug entry: Treatments that last more than 1 day 
        drug_interval_1 = treat_dates[0:2] # For the first drug combination
        treatment_time_1 = drug_interval_1[1] - drug_interval_1[0]
        #print(treatment_time_1)
        drugs_1 = np.array(row['MMTX_THERAPY'])
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
                par1.add_patch(Rectangle((drug_interval_1[0], drugkey - plotheight/2), treatment_time_1, plotheight, zorder=2, color=drug_colordict[drugkey]))

    ########################################################################################################################################################################

    # end and save plot:
    host.set_title("Patient ID " + str(PUBLIC_ID))
    host.set_xlabel("Time (year)")
    host.set_ylabel("Serum Mprotein (g/dL)")
    host.set_ylim(bottom=0)
    par1.set_ylabel("Drug")
    par1.set_yticks(range(maxdrugkey+1))
    par1.set_yticklabels(range(maxdrugkey+1))
    #par1.set_ylim([-0.5,len(unique_drugs)+0.5]) # If you want to cover all unique drugs
    host.set_zorder(host.get_zorder()+3)
    #host.autofmt_xdate()
    #plt.tight_layout()
    # Only save the plot if there are at least 3 M protein measurements
    inclusion_criteria_satisfied = count_mprotein > 2 # and count_treatments > 0
    print(PUBLIC_ID)
    plt.draw()
    plt.show()
    if inclusion_criteria_satisfied:
        plt.savefig("./COMMPASS_Mproteinplots/" + str(PUBLIC_ID) + ".png")
    plt.close()
    return [count_mprotein, count_treatments, inclusion_criteria_satisfied]

# Plot them all and capture the M protein counts in the same line
# patient output: [count_mprotein, count_treatments, inclusion_criteria_satisfied]
patient_output = [plot_patient(this_PUBLIC_ID) for this_PUBLIC_ID in unique_PUBLIC_IDs[0:3]]
counts_mprotein, counts_treatments, counts_inclusion_criteria_satisfied = [[pat_out[iii] for pat_out in patient_output] for iii in [0,1,2]]

print("There are "+str(sum(counts_inclusion_criteria_satisfied))+" patients that satisfy the criteria for inclusion.")

# Histogram of number of M protein measurements 
plt.figure()
plt.hist(counts_mprotein, bins = max(counts_mprotein)+1) #mprotein counts
plt.title("Histogram of number of M protein measurements ") #, y=1.02)
plt.xlabel("Number of M protein measurements") #, labelpad=14)
plt.ylabel("Count") #, labelpad=14)
plt.xticks()
plt.savefig("./COMMPASS_Mprotein_count_histogram.png")
plt.show()
plt.close()


"""
data_treat_line_1 = df_treatment_lines[["Treatment line 1"]]
sns.set(font_scale=0.6)
data_treat_line_1.value_counts().plot(kind='bar', figsize=(7, 6), rot=0)

#################################################################################################################
# Treatment lines and M protein history plot
#################################################################################################################
# For each patient (PUBLIC_ID), we make a figure and plot the treatment lines
treat_colordict = dict(zip(treatment_line_ids, treat_line_colors))
# Initialize PUBLIC_ID
PUBLIC_ID = df_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
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
    # If it's a new patient, then save plot and initialize new figure with new PUBLIC_ID
    if not (df_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
        ax1.set_title("Patient ID " + str(PUBLIC_ID))
        ax1.set_xlabel("Time (year)")
        ax1.set_ylabel("Serum Mprotein (g/dL)")
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
            plt.savefig("./COMMPASS_Treatment_line_plots/" + str(PUBLIC_ID) + ".pdf")
        #plt.show()
        plt.close()

        PUBLIC_ID = df_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]
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
        treat_line_id = treatment_to_id_dictionary[this_treatment_line]
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

ax1.set_title("Patient ID " + str(PUBLIC_ID))
ax1.set_xlabel("Time (year)")
ax1.set_ylabel("Serum Mprotein (g/dL)")
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
    plt.savefig("./COMMPASS_Treatment_line_plots/" + str(PUBLIC_ID) + "_COMMPASS.pdf")
#plt.show()
plt.close()

#################################################################################################################
# Plot matrix of drug counts per patient
#################################################################################################################
# dictionary of drugs: drug_dictionary = dict(zip(unique_drugs, drug_ids))
unique_PUBLIC_IDs = pd.unique(df_mprotein_and_dates[['PUBLIC_ID']].values.ravel('K'))
print(unique_PUBLIC_IDs)
PUBLIC_ID_dict_keys = range(len(unique_PUBLIC_IDs))
PUBLIC_ID_dictionary = dict(zip(unique_PUBLIC_IDs, PUBLIC_ID_dict_keys))
number_of_PUBLIC_IDs = len(unique_PUBLIC_IDs)
print("There are "+str(number_of_PUBLIC_IDs)+" unique PUBLIC_IDs.")
#%matplotlib inline
Index= unique_PUBLIC_IDs
Cols = unique_drugs
drugmatrix = DataFrame(np.zeros((number_of_PUBLIC_IDs,number_of_unique_drugs)), index=Index, columns=Cols)

# Loop over rows. For each line, find the drug names, lookup in dict and update matrix 
# Initialize PUBLIC_ID
PUBLIC_ID = df_mprotein_and_dates.loc[1,['PUBLIC_ID']][0]
PUBLIC_ID_key = PUBLIC_ID_dictionary[PUBLIC_ID] # The dataframe row index
for row_index in range(len(df_mprotein_and_dates)):
    # Check if it's the same patient.
    if not (df_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0] == PUBLIC_ID):
        PUBLIC_ID = df_mprotein_and_dates.loc[row_index,['PUBLIC_ID']][0]
        PUBLIC_ID_key = PUBLIC_ID_dictionary[PUBLIC_ID] # The dataframe row index

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
            drugmatrix.iloc[PUBLIC_ID_key,drugkey] = drugmatrix.iloc[PUBLIC_ID_key,drugkey] + 1 

    drug_interval_2 = treat_dates[2:4] # For the second drug combination
    drugs_2 = np.array(df_mprotein_and_dates.loc[row_index, ['Drug 1.1', 'Drug 2.1', 'Drug 3.1', 'Drug 4.1']])
    # Remove cases with missing end dates
    missing_date_bool = isNaN(drug_interval_2).any()
    if not missing_date_bool: 
        # Remove nan drugs
        drugs_2 = drugs_2[~isNaN(drugs_2)]
        for ii in range(len(drugs_2)):
            drugkey = drug_dictionary[drugs_2[ii]]
            drugmatrix.iloc[PUBLIC_ID_key,drugkey] = drugmatrix.iloc[PUBLIC_ID_key,drugkey] + 1 

# Count matrix showing how many times drug was given to patient
plt.figure(figsize=(15,15))
sns.heatmap(drugmatrix.iloc[:,:], annot=False)
plt.tight_layout()
plt.savefig("./drug_matrix_COMMPASS.pdf")
#plt.show()

# Binary matrix indicating whether the drug was used
plt.figure(figsize=(15,15))
sns.heatmap(drugmatrix.iloc[:,:]>0, annot=False)
plt.tight_layout()
plt.savefig("./drug_matrix_binary_COMMPASS.pdf")
#plt.show()
"""
