# Purposes of this script: 
#   Load COMMPASS patient data, create:
#   - COMMPASS_patient_dictionary
#   - treatment_to_id_dictionary_COMMPASS 
#   - df_mprotein_and_dates
#   - df_drugs_and_dates
from utilities import *

# Load things
# M protein data
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_VISIT_V2.tsv'
df = pd.read_csv(filename, sep='\t')
df_mprotein_and_dates = df[['PUBLIC_ID', 'VISITDY',
'D_LAB_serum_m_protein', 'D_LAB_serum_kappa', 'D_LAB_serum_lambda',
#'D_IM_LIGHT_CHAIN_BY_FLOW', #"kappa" or "lambda"
#'D_LAB_serum_kappa', # Serum Kappa (mg/dL)
#'D_LAB_serum_lambda', # Serum Lambda (mg/dL)
#'D_IM_kaplam'
]]

# Drop nan measurements 
df_mprotein_and_dates = df_mprotein_and_dates.dropna(subset=['PUBLIC_ID', 'VISITDY', 'D_LAB_serum_m_protein'], inplace=False)

# Impute nan values for Kappa and Lambda 
values = {'D_LAB_serum_kappa':0, 'D_LAB_serum_lambda':0}
df_mprotein_and_dates = df_mprotein_and_dates.fillna(value=values)

# Here is a dictionary of patients indexed by their names: Name (str) --> Patient object (Patient)
COMMPASS_patient_dictionary = {}

## Only patient 1143 for testing
#df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['PUBLIC_ID'] == 'MMRF_1293'] # 'MMRF_1293' another with 6,7,8 ; 'MMRF_1256' for only drug 8

print("Iterating the M protein file and making patients of all the entries")
# Iterate the M protein file and make patients of all the entries
for index, row in df_mprotein_and_dates.iterrows():
    patient_name = row['PUBLIC_ID']
    if not patient_name in COMMPASS_patient_dictionary.keys():
        COMMPASS_patient_dictionary[patient_name] = COMMPASS_Patient(measurement_times=np.array([row['VISITDY']]), drug_dates=set(), drug_history=np.array([]), treatment_history=np.array([]), Mprotein_values=np.array([row['D_LAB_serum_m_protein']]), Kappa_values=np.array([row['D_LAB_serum_kappa']]), Lambda_values=np.array([row['D_LAB_serum_lambda']]), covariates=np.array([]), name=patient_name)
    else:
        COMMPASS_patient_dictionary[patient_name].add_Mprotein_line_to_patient(row['VISITDY'], row['D_LAB_serum_m_protein'], row['D_LAB_serum_kappa'], row['D_LAB_serum_lambda'])

print("Adding drugs")
# Add drugs 
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_STAND_ALONE_TREATMENT_REGIMEN_V2.tsv'
df = pd.read_csv(filename, sep='\t')
df_drugs_and_dates = df[[
    'PUBLIC_ID', 'MMTX_THERAPY', 
    'startday', 'stopday'
]]

# Drop nan measurements 
df_drugs_and_dates = df_drugs_and_dates.dropna(subset=['PUBLIC_ID', 'MMTX_THERAPY', 'startday', 'stopday'], inplace=False)

## Only patient 1143 for testing
#df_drugs_and_dates = df_drugs_and_dates[df_drugs_and_dates['PUBLIC_ID'] == 'MMRF_1293'] # 'MMRF_1293' another with 6,7,8 ; 'MMRF_1256' for only drug 8

df_drugs_and_dates['drug_id'] = np.nan
for index, row in df_drugs_and_dates.iterrows():
    if not isNaN(row['MMTX_THERAPY']): 
        patient_name = row['PUBLIC_ID']
        drug_name = row['MMTX_THERAPY']
        drug_id = drug_dictionary[drug_name]
        # Temporary: Set drug id (continuous variable for feature extraction) for patients with that drug
        df_drugs_and_dates.loc[(df_drugs_and_dates["MMTX_THERAPY"] == drug_name),'drug_id'] = drug_id
        drug_period_object = Drug_period(row['startday'], row['stopday'], drug_id)
        if not patient_name in COMMPASS_patient_dictionary.keys():
            # Add patient with drugs but no M protein measurements
            COMMPASS_patient_dictionary[patient_name] = COMMPASS_Patient(measurement_times=np.array([]), drug_dates=set(), drug_history=np.array([drug_period_object]), treatment_history=np.array([]), Mprotein_values=np.array([]), Kappa_values=np.array([]), Lambda_values=np.array([]), covariates=np.array([]), name=patient_name)
        else:
            COMMPASS_patient_dictionary[patient_name].add_drug_period_to_patient(drug_period_object)

treatment_to_id_dictionary_COMMPASS = {}
unique_treatment_lines = []
unique_treat_counter = 0
# Take drug histories, turn them into treatment histories
# At the same time, we build the treatment dictionary (drug_set --> treatment_id) of the COMMPASS dataset
for patient_name, patient_object in COMMPASS_patient_dictionary.items():
    # The treatment changes whenever a drug starts or stops
    date_set = patient_object.drug_dates
    # also, the first and last M protein measurements can be the first and last time point in the patient history
    if len(date_set) > 0:
        if len(patient_object.measurement_times) > 0:
            if min(patient_object.measurement_times) < min(date_set):
                date_set.add(min(patient_object.measurement_times))
            if max(patient_object.measurement_times) > max(date_set):
                date_set.add(max(patient_object.measurement_times))
        # Iterate through the date set and keep track of which drugs are being turned on and off 
        active_drug_set = set()
        prev_date = -np.inf
        for index, this_date in enumerate(sorted(date_set)): # the sorted set is a list
            if index > 0:
                # Update patient history with the treatment that ends on this date
                frozen_drug_set = frozenset(active_drug_set) # Frozen sets are hashable and allows lookup in a dictionary
                # Check if the treatment combination is a new one
                if frozen_drug_set not in unique_treatment_lines:
                    # Keep track of the unique treatment lines that patients are exposed to 
                    unique_treatment_lines.append(frozen_drug_set)
                    # Then unique_treat_counter is the id of this treatment. We add it to the dictionary
                    treatment_to_id_dictionary_COMMPASS[frozen_drug_set] = unique_treat_counter
                    # Add it to the patient history
                    patient_object.add_treatment_to_treatment_history(Treatment(prev_date, this_date, unique_treat_counter))
                    unique_treat_counter = unique_treat_counter + 1
                else: # Add the id of this treatment combination to the history
                    patient_object.add_treatment_to_treatment_history(Treatment(prev_date, this_date, treatment_to_id_dictionary_COMMPASS[frozen_drug_set]))
            # Find the drugs that are being turned off at this date
            for drug_period in patient_object.drug_history:
                if drug_period.end == this_date:
                    active_drug_set.discard(drug_period.id)
            # Find the drugs that are being turned on 
            for drug_period in patient_object.drug_history:
                if drug_period.start == this_date:
                    active_drug_set.add(drug_period.id)
            prev_date = this_date
            # Now we are in the interval following this_date, and we know which drugs are active here. 
        # At the end of the loop all drugs are turned off
treatment_id_to_drugs_dictionary_COMMPASS = {v: k for k, v in treatment_to_id_dictionary_COMMPASS.items()}
# Save it here
picklefile = open('./binaries_and_pickles/treatment_id_to_drugs_dictionary_COMMPASS', 'wb')
pickle.dump(treatment_id_to_drugs_dictionary_COMMPASS, picklefile)
picklefile.close()

print("len(COMMPASS_patient_dictionary): ", len(COMMPASS_patient_dictionary))
# What is the most common treatment? 
print("There are ", unique_treat_counter, " unique treatments")
# For each treatment, how many patients got that treatment? 
count_patients_per_treatment = np.zeros(len(treatment_to_id_dictionary_COMMPASS))
for treatment_id in treatment_to_id_dictionary_COMMPASS.values():
    for name, patient in COMMPASS_patient_dictionary.items():
        if treatment_id in [treatment.id for treatment in patient.treatment_history]:
            count_patients_per_treatment[treatment_id] = count_patients_per_treatment[treatment_id] + 1

# For each treatment, how many regions are there? 
count_regions_per_treatment = np.zeros(len(treatment_to_id_dictionary_COMMPASS))
for name, patient in COMMPASS_patient_dictionary.items():
    for treatment in patient.treatment_history:
        count_regions_per_treatment[treatment.id] = count_regions_per_treatment[treatment.id] + 1

def Sort(sub_li):
    return(sorted(sub_li, key = lambda x: x[1]))    
# Print most popular treatments
## per number of patients
print("Sort(enumerate(count_patients_per_treatment))") #, Sort(enumerate(count_patients_per_treatment)))
counts_patients = np.flip(Sort(enumerate(count_patients_per_treatment)))
for element in counts_patients[counts_patients[:,0]>100]:
    patient_count = element[0]
    treatment_id = element[1]
    drug_names = get_drug_names_from_treatment_id_COMMPASS(treatment_id, treatment_id_to_drugs_dictionary_COMMPASS)
    print("N =", patient_count, "   id = ", treatment_id, "    drugs:", drug_names)

## per number of regions
print("Sort(enumerate(count_regions_per_treatment))") #, Sort(enumerate(count_regions_per_treatment)))
counts_regions = np.flip(Sort(enumerate(count_regions_per_treatment)))
for element in counts_regions[counts_regions[:,0]>100]:
    patient_count = element[0]
    treatment_id = element[1]
    drug_names = get_drug_names_from_treatment_id_COMMPASS(treatment_id, treatment_id_to_drugs_dictionary_COMMPASS)
    print("N =", patient_count, "   id = ", treatment_id, "    drugs:", drug_names)

# Show id of drugs 
print('Melphalan: drug id', drug_dictionary_COMMPASS['Melphalan'])
print('Cyclophosphamide: drug id', drug_dictionary_COMMPASS['Cyclophosphamide'])
print('Dexamethasone: drug id', drug_dictionary_COMMPASS['Dexamethasone'])
print('Bortezomib: drug id', drug_dictionary_COMMPASS['Bortezomib'])
print('Lenalidomide: drug id', drug_dictionary_COMMPASS['Lenalidomide'])

count_patients_per_treatment = count_patients_per_treatment #[:100]
plt.figure()
plt.plot(np.flip(sorted(count_patients_per_treatment[1:])))
plt.title("Number of patients that got each treatment")
plt.xlabel("Treatment") #, labelpad=14)
plt.ylabel("Number of patients") #, labelpad=14)
plt.xticks()
plt.savefig("./plots/count_patients_per_treatment.png")
#plt.show()
plt.close()

count_regions_per_treatment = count_regions_per_treatment #[:100]
plt.figure()
plt.plot(np.flip(sorted(count_regions_per_treatment[1:])))
plt.title("Number of regions for each treatment")
plt.xlabel("Treatment") #, labelpad=14)
plt.ylabel("Number of patients") #, labelpad=14)
plt.xticks()
plt.savefig("./plots/count_regions_per_treatment.png")
#plt.show()
plt.close()
COMMPASS_patient_dictionary["MMRF_1143"].print()


# Save things
# Save (training_instance_id, PUBLIC_ID, [start, end]) to allow df selection out of df_mprotein and df_drugs for feature extraction indexed by training_instance_id
###df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['training_instance_id'].notna()]
df_mprotein_and_dates.reset_index(drop=True, inplace=True)
df_mprotein_and_dates.to_pickle("./binaries_and_pickles/df_mprotein_and_dates.pkl")

###df_drugs_and_dates = df_drugs_and_dates[df_drugs_and_dates['training_instance_id'].notna()]
df_drugs_and_dates.reset_index(drop=True, inplace=True)
df_drugs_and_dates.to_pickle("./binaries_and_pickles/df_drugs_and_dates.pkl")

picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'wb')
pickle.dump(COMMPASS_patient_dictionary, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/treatment_to_id_dictionary_COMMPASS', 'wb')
pickle.dump(treatment_to_id_dictionary_COMMPASS, picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/unique_treat_counter', 'wb')
pickle.dump(np.array(unique_treat_counter), picklefile)
picklefile.close()
