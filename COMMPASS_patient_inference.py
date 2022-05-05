# Purposes of this script: 
#   Load COMMPASS patient data, create treatment_to_id_dictionary_COMMPASS
#   Find sections with the right drug combination and enough data to perform inference
#   Perform inference on parameter set
#   Feature extraction of patient history
#   Learn mapping from extracted features to parameter set 
from utilities import *

# M protein data
filename = './COMMPASS_data/CoMMpass_IA17_FlatFiles/MMRF_CoMMpass_IA17_PER_PATIENT_VISIT_V2.tsv'
df = pd.read_csv(filename, sep='\t')
df_mprotein_and_dates = df[['PUBLIC_ID', 'VISITDY',
'D_LAB_serum_m_protein',
#'D_IM_LIGHT_CHAIN_BY_FLOW', #"kappa" or "lambda"
#'D_LAB_serum_kappa', # Serum Kappa (mg/dL)
#'D_LAB_serum_lambda', # Serum Lambda (mg/dL)
#'D_IM_kaplam'
]]

# Drop nan measurements 
df_mprotein_and_dates = df_mprotein_and_dates.dropna(subset=['PUBLIC_ID', 'VISITDY', 'D_LAB_serum_m_protein'], inplace=False)

# Here is a dictionary of patients indexed by their names: Name (str) --> Patient object (Patient)
COMMPASS_patient_dictionary = {}

## Only patient 1143 for testing
#df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['PUBLIC_ID'] == 'MMRF_1293'] # 'MMRF_1293' another with 6,7,8 ; 'MMRF_1256' for only drug 8

# Iterate the M protein file and make patients of all the entries
for index, row in df_mprotein_and_dates.iterrows():
    patient_name = row['PUBLIC_ID']
    if not patient_name in COMMPASS_patient_dictionary.keys():
        COMMPASS_patient_dictionary[patient_name] = COMMPASS_Patient(measurement_times=np.array([row['VISITDY']]), drug_dates=set(), drug_history=np.array([]), treatment_history=np.array([]), observed_values=np.array([row['D_LAB_serum_m_protein']]), covariates=np.array([]), name=patient_name)
    else:
        COMMPASS_patient_dictionary[patient_name].add_Mprotein_line_to_patient(row['VISITDY'], row['D_LAB_serum_m_protein'])

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

for index, row in df_drugs_and_dates.iterrows():
    if not isNaN(row['MMTX_THERAPY']): 
        patient_name = row['PUBLIC_ID']
        drug_name = row['MMTX_THERAPY']
        drug_id = drug_dictionary[drug_name]
        drug_period_object = Drug_period(row['startday'], row['stopday'], drug_id)
        if not patient_name in COMMPASS_patient_dictionary.keys():
            # Add patient with drugs but no M protein measurements
            COMMPASS_patient_dictionary[patient_name] = COMMPASS_Patient(measurement_times=np.array([]), drug_dates=set(), drug_history=np.array([drug_period_object]), treatment_history=np.array([]), observed_values=np.array([]), covariates=np.array([]), name=patient_name)
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
def get_drug_names_from_treatment_id(treatment_id):
    drug_set = treatment_id_to_drugs_dictionary_COMMPASS[treatment_id]
    drug_names = [drug_id_to_name_dictionary[elem] for elem in drug_set]
    return drug_names

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

print("Done counting stuff")
def Sort(sub_li):
    return(sorted(sub_li, key = lambda x: x[1]))    
# Most popular treatments
## per number of patients
print("Sort(enumerate(count_patients_per_treatment))", Sort(enumerate(count_patients_per_treatment)))
counts_patients = np.flip(Sort(enumerate(count_patients_per_treatment)))
for element in counts_patients[counts_patients[:,0]>100]:
    patient_count = element[0]
    treatment_id = element[1]
    drug_names = get_drug_names_from_treatment_id(treatment_id)
    print("N =", patient_count, "   id = ", treatment_id, "    drugs:", drug_names)

## per number of regions
print("Sort(enumerate(count_regions_per_treatment))", Sort(enumerate(count_regions_per_treatment)))
counts_regions = np.flip(Sort(enumerate(count_regions_per_treatment)))
for element in counts_regions[counts_regions[:,0]>100]:
    patient_count = element[0]
    treatment_id = element[1]
    drug_names = get_drug_names_from_treatment_id(treatment_id)
    print("N =", patient_count, "   id = ", treatment_id, "    drugs:", drug_names)

count_patients_per_treatment = count_patients_per_treatment #[:100]
plt.figure()
plt.plot(np.flip(sorted(count_patients_per_treatment[1:])))
plt.title("Number of patients that got each treatment")
plt.xlabel("Treatment") #, labelpad=14)
plt.ylabel("Number of patients") #, labelpad=14)
plt.xticks()
plt.savefig("./count_patients_per_treatment.png")
#plt.show()
plt.close()

count_regions_per_treatment = count_regions_per_treatment #[:100]
plt.figure()
plt.plot(np.flip(sorted(count_regions_per_treatment[1:])))
plt.title("Number of regions for each treatment")
plt.xlabel("Treatment") #, labelpad=14)
plt.ylabel("Number of patients") #, labelpad=14)
plt.xticks()
plt.savefig("./count_regions_per_treatment.png")
#plt.show()
plt.close()

# Find the treatment id of the required treatment, extract those treatment regions and perform inference there
#COMMPASS_patient_dictionary["MMRF_1143"].print()
treatment_id_of_interest = 15 # Dex+Len+Bor #COMMPASS_patient_dictionary["MMRF_1293"].treatment_history[5].id
###how_many_regions = np.zeros(unique_treat_counter)
###for treatment_id_of_interest in range(unique_treat_counter):
#print("Treatment id of interest:", treatment_id_of_interest)

## Inference
# Simple exponential growth model with 2 populations, where only one is affected by treatment
# The parameters we estimate are 
#               Y_0, pi_r,   g_r,   g_s,  k_1,  sigma
lb = np.array([  0,    0,  0.00,  0.00, 0.00]) #, 10e-6])
ub = np.array([100,    1,  2e-1,  1e-1, 1e-0]) #, 10e4])
# Y_0=50, pi_r=0.10, g_r=2e-3, g_s=1e-2, k_1=3e-2

# A training instance is a pair of history covariates X and estimated parameters Y
# Define minimum number of measurements for including period as training instance to X and Y
minimum_number_of_measurements = 3
number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id = 0
print("\nFinding right regions and estimating parameters...")
# Iterate through patients
# Identify valid periods and estimate parameters there: Y: (training_instance_id, parameters) X_periods = (training_instance_id, [start, end])
# Find patients with periods that satisfy criteria of enough M protein measurements under chosen treatment 
# 1) Feature extract history up until period, put in X
# 2) Estimate parameters in period, put in Y
training_instance_id = 0
df_mprotein_and_dates['training_instance_id'] = np.nan
X_periods = {}
Y = []
N_iter = 10000
this_history = np.array([])
for name, patient in COMMPASS_patient_dictionary.items():
    # Find periods of interest by looking through patient history 
    period_start = np.nan
    period_end = np.nan
    valid_interval = False
    for index, treatment in enumerate(patient.treatment_history):
        if valid_interval == False: 
            if treatment.id == treatment_id_of_interest:
                # We found the start 
                #print("We found the start ")
                valid_interval = True
                period_start = treatment.start
                this_history = np.array([treatment])
                # Continute to next
            else: 
                # Continue looking for the start
                pass 
        else: # if valid interval == True, then we are looking for 0 or the end 
            if treatment.id == treatment_id_of_interest:
                # Only if two back-to-back cases of correct id. This should not happen
                this_history = np.append(this_history,[treatment])
                pass
            elif treatment.id == 0:
                # Extending the period by a drug holiday, then it's the end at the end of treatment id 0
                #print("Extending the period by a drug holiday, then ending this period")
                this_history = np.append(this_history,[treatment])
                valid_interval = False
                period_end = treatment.end
                # Check how many M protein values are within period, and find the observed values, times and drug periods in the period
                valid_observations = patient.observed_values[patient.measurement_times>=period_start]
                valid_times = patient.measurement_times[patient.measurement_times>=period_start]
                valid_observations = valid_observations[valid_times<=period_end]
                valid_times = valid_times[valid_times<=period_end]
                # Only add as data instance to X and Y if there are enough valid times
                if len(valid_times) >= minimum_number_of_measurements and min(valid_observations) > 0:
                    number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id = number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id + 1
                    # Note the limits of this period
                    X_periods[training_instance_id] = [patient.name, period_start, period_end] # optional
                    # Set training_instance_id for slice with correct patient id + M protein measurements within period:
                    df_mprotein_and_dates[(df_mprotein_and_dates["PUBLIC_ID"] == patient.name) & (df_mprotein_and_dates["VISITDY"] > period_start) & (df_mprotein_and_dates["VISITDY"] < period_end)]['training_instance_id'] = training_instance_id
                    # Estimate parameters for a dummy patient within this interval
                    dummmy_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=this_history, observed_values=valid_observations, covariates=[], name="dummy")
                    this_estimate = estimate_drug_response_parameters(dummmy_patient, lb, ub, N_iterations=N_iter)
                    # Add estimates to Y
                    Y.append(this_estimate) # training_instance_id is position in Y
                    patient.add_parameter_estimate(this_estimate, (period_start, period_end))
                    training_instance_id = training_instance_id + 1
            else: 
                # We found the end at the beginning of this foreign treatment id
                valid_interval = False
                period_end = treatment.start
                # Check how many M protein values are within period, and find the observed values, times and drug periods in the period
                valid_observations = patient.observed_values[patient.measurement_times>=period_start]
                valid_times = patient.measurement_times[patient.measurement_times>=period_start]
                valid_observations = valid_observations[valid_times<=period_end]
                valid_times = valid_times[valid_times<=period_end]
                # Only add as data instance to X and Y if there are enough:
                if len(valid_times) >= minimum_number_of_measurements and min(valid_observations) > 0:
                    number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id = number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id + 1
                    # Note the limits of this period
                    X_periods[training_instance_id] = [patient.name, period_start, period_end] # optional
                    # Set training_instance_id for slice with correct patient id + M protein measurements within period:
                    df_mprotein_and_dates[(df_mprotein_and_dates["PUBLIC_ID"] == patient.name) & (df_mprotein_and_dates["VISITDY"] > period_start) & (df_mprotein_and_dates["VISITDY"] < period_end)]['training_instance_id'] = training_instance_id
                    # Estimate parameters for a dummy patient within this interval
                    dummmy_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=this_history, observed_values=valid_observations, covariates=[], name="dummy")
                    this_estimate = estimate_drug_response_parameters(dummmy_patient, lb, ub, N_iterations=N_iter)
                    # Add estimates to Y
                    Y.append(this_estimate) # training_instance_id is position in Y
                    patient.add_parameter_estimate(this_estimate, (period_start, period_end))
                    training_instance_id = training_instance_id + 1
    # After the last treatment, if ending on a valid interval, this is the end
    if valid_interval == True: 
        #print("The end was the end of the last treatment")
        period_end = patient.treatment_history[-1].end
        # Check how many M protein values are within period, and find the observed values, times and drug periods in the period
        #print(period_start, period_end)
        valid_observations = patient.observed_values[patient.measurement_times>=period_start]
        valid_times = patient.measurement_times[patient.measurement_times>=period_start]
        valid_observations = valid_observations[valid_times<=period_end]
        valid_times = valid_times[valid_times<=period_end]
        # Only add as data instance to X and Y if there are enough:
        if len(valid_times) >= minimum_number_of_measurements and min(valid_observations) > 0:
            number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id = number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id + 1
            # Note the limits of this period
            X_periods[training_instance_id] = [patient.name, period_start, period_end] # optional
            # Set training_instance_id for slice with correct patient id + M protein measurements within period:
            df_mprotein_and_dates[(df_mprotein_and_dates["PUBLIC_ID"] == patient.name) & (df_mprotein_and_dates["VISITDY"] > period_start) & (df_mprotein_and_dates["VISITDY"] < period_end)]['training_instance_id'] = training_instance_id
            # Estimate parameters for a dummy patient within this interval
            dummmy_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=this_history, observed_values=valid_observations, covariates=[], name="dummy")
            this_estimate = estimate_drug_response_parameters(dummmy_patient, lb, ub, N_iterations=N_iter)
            # Add estimates to Y
            Y.append(this_estimate) # training_instance_id is position in Y
            patient.add_parameter_estimate(this_estimate, (period_start, period_end))
            training_instance_id = training_instance_id + 1
    # Plotting estimated parameters
    for index, param_set in enumerate(patient.parameter_estimates):
        # Find parameter set and corresponding period
        estimated_parameters = patient.parameter_estimates[index]
        period_start, period_end = [patient.parameter_periods[index][ii] for ii in [0,1]]
        # Find valid observation at valid times:
        valid_observations = patient.observed_values[patient.measurement_times>=period_start]
        valid_times = patient.measurement_times[patient.measurement_times>=period_start]
        valid_observations = valid_observations[valid_times<=period_end]
        valid_times = valid_times[valid_times<=period_end]
        # Take the part of the history that is within the period of interest
        valid_history = np.array([])
        for treatment in patient.treatment_history:
            if treatment.start >= period_start and treatment.end <= period_end:
                valid_history = np.append(valid_history,[treatment])
        plot_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=valid_history, observed_values=valid_observations, covariates=[], name=patient.name)
        savename = "./COMMPASS_estimate_plots/treatment_id_"+str(treatment_id_of_interest)+"/Treatment_"+str(treatment_id_of_interest)+"_"+patient.name+"Y_0="+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sigma="+str(estimated_parameters.sigma)+".png"
        plot_title = patient.name #+"\nY_0="+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sig=", str(estimated_parameters.sigma)
        plot_treatment_region_with_estimate(estimated_parameters, plot_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=plot_title, savename=savename)

#COMMPASS_patient_dictionary["MMRF_1293"].print()

print("Treatment id of interest:", treatment_id_of_interest)
print("Number of regions with", minimum_number_of_measurements, "or more M protein measurements:", number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id)

###how_many_regions[treatment_id_of_interest] = number_of_regions_with_at_least_minimum_number_of_measurements_for_this_treatment_id
####print("Sort(enumerate(how_many_regions))", Sort(enumerate(how_many_regions)))
###counts_regions = np.flip(Sort(enumerate(how_many_regions)))
###for element in counts_regions[counts_regions[:,0]>1]:
###    patient_count = element[0]
###    treatment_id = element[1]
###    drug_names = get_drug_names_from_treatment_id(treatment_id)
###    print("N =", patient_count, "   id = ", treatment_id, "    drugs:", drug_names)

# Save (training_instance_id, PUBLIC_ID, [start, end]) to allow df selection out of df_mprotein and df_drugs for feature extraction indexed by training_instance_id
#df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['training_instance_id'].notna()]
#df_mprotein_and_dates.reset_index(drop=True, inplace=True)
#from tsfresh import extract_features
#df_mprotein_and_dates = df_mprotein_and_dates[["training_instance_id", "D_LAB_serum_m_protein", "VISITDY"]]
#extracted_features = extract_features(df_mprotein_and_dates, column_id="training_instance_id", column_sort="VISITDY")

# Do the machine learning magic with X and Y
#Y = np.array(Y)

# Plot estimate with observed values
#estimated_parameters_patient_2 = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=global_sigma)
#plot_treatment_region_with_estimate(parameters_patient_2, patient_2, estimated_parameters=estimated_parameters_patient_2, PLOT_ESTIMATES=True, plot_title="Patient 2")



