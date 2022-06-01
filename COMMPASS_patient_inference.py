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
picklefile = open('treatment_id_to_drugs_dictionary_COMMPASS', 'wb')
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
COMMPASS_patient_dictionary["MMRF_1143"].print()

def get_binary_outcome(period_start, patient, this_estimate):
    # NB! To predict chances under received treatment, we must encode future treatment precisely in covariates to predict effect of future treatment.  This means encode treatment+drug holiday in X.
    # What we do now is to predict outcome under continuous administration of treatment, for the time interval where we predict response.  

    # Using estimated Y_0 as M protein value at treatment start
    initial_Mprotein_value = this_estimate.Y_0
    
    # If projected/observed M protein value goes above M protein value at treatment start within X days, then outcome = 1
    days_for_consideration = 182 # Window from treatment start within which we check for increase

    future_starts = np.array([elem.start for elem in patient.treatment_history])
    #future_treatments = patient.treatment_history[future_starts >= period_start]
    #future_treatments = np.array([Treatment(elem.start - period_start, elem.end - period_start, elem.id) for elem in future_treatments])
    first_future_treatment = patient.treatment_history[future_starts >= period_start][0]
    future_treatments = np.array([Treatment(period_start, period_start + days_for_consideration, first_future_treatment.id)])

    # Using predicted Mprotein value to check for increase
    predicted_Mprotein = measure_Mprotein_noiseless(this_estimate, np.array([period_start+days_for_consideration]), future_treatments)
    if predicted_Mprotein[0] > initial_Mprotein_value:
        return 1
    # Return nan if the future history is shorter than the period we consider
    elif max(np.array([elem.end for elem in future_treatments]) < days_for_consideration):
        return np.nan
    else: # If we did not observe relapse and the observed period is long enough, report 0
        return 0
    ## Using observed values to check for increase
    #future_times = patient.measurement_times[patient.measurement_times > period_start]
    #future_Mprotein = patient.Mprotein_values[patient.measurement_times > period_start]
    #relevant_times = future_times[future_times <= (period_start+days_for_consideration)]
    #relevant_Mprotein = future_Mprotein[future_times <= (period_start+days_for_consideration)]
    #
    #if len(relevant_Mprotein[relevant_Mprotein > initial_Mprotein_value]) > 0:
    #    binary_outcome = 1
    #elif len(relevant_Mprotein[relevant_Mprotein <= initial_Mprotein_value]) > 0:
    #    binary_outcome = 0
    #else: # if the list has no length, i.e. we have no measurements of it
    #    binary_outcome = np.nan
    #return binary_outcome

# Find the treatment id of the required treatment, extract those treatment regions and perform inference there
def estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, Y_increase_or_not):
    # Check how many M protein values are within period, and find the observed values, times and drug periods in the period
    valid_Mprotein = dummy_Mprotein_values[dummy_measurement_times>=period_start]
    valid_Kappa = dummy_Kappa_values[dummy_measurement_times>=period_start]
    valid_Lambda = dummy_Lambda_values[dummy_measurement_times>=period_start]
    valid_times = dummy_measurement_times[dummy_measurement_times>=period_start]
    valid_Mprotein = valid_Mprotein[valid_times<=period_end]
    valid_Kappa = valid_Kappa[valid_times<=period_end]
    valid_Lambda = valid_Lambda[valid_times<=period_end]
    valid_times = valid_times[valid_times<=period_end]
    # Only add as data instance to X and Y if there are enough:
    if len(valid_times) >= minimum_number_of_measurements and min(valid_Mprotein) > 0:
        print("Saving a case from", patient.name, "- treatment id", treatment_id_of_interest)
        how_many_regions[treatment_id_of_interest] = how_many_regions[treatment_id_of_interest] + 1
        # Note the time limits of this period
        training_instance_dict[training_instance_id] = [patient.name, period_start, period_end, treatment_id_of_interest]
        # Estimate parameters for a dummy patient within this interval
        dummmy_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=this_history, Mprotein_values=valid_Mprotein, Kappa_values=valid_Kappa, Lambda_values=valid_Lambda, covariates=[], name="dummy")
        this_estimate = estimate_drug_response_parameters(dummmy_patient, lb, ub, N_iterations=N_iter)
        # Add estimates to Y_parameters
        Y_parameters.append(this_estimate) # training_instance_id is position in Y_parameters
        binary_outcome = get_binary_outcome(period_start, patient, this_estimate)
        Y_increase_or_not = np.concatenate((Y_increase_or_not, np.array([binary_outcome])))
        patient.add_parameter_estimate(this_estimate, (period_start, period_end), dummmy_patient)
        training_instance_id = training_instance_id + 1

        # Plotting treatment region with estimate
        #for index, param_set in enumerate(patient.parameter_estimates):
        estimated_parameters = this_estimate # patient.parameter_estimates[index]
        plot_patient = dummmy_patient # patient.dummmy_patients[index]
        savename = "./COMMPASS_estimate_plots/treatment_id_"+str(treatment_id_of_interest)+"/Treatment_"+str(treatment_id_of_interest)+"_"+patient.name+"_at_time"+str(period_start)+"Y_0="+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sigma="+str(estimated_parameters.sigma)+".png"
        plot_title = patient.name
        plot_treatment_region_with_estimate(estimated_parameters, plot_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=plot_title, savename=savename)
    return training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, Y_increase_or_not

## Inference
# The length of ub and lb implicitly decides whether the effect of treatment is given a parameter or not. 
# If len(ub) = 4, then it is assumed that only periods under treatment are considered
# If len(ub) = 5, then k_1 models the effect of the drug on the sensitive population
# Simple exponential growth model with 2 populations, where only one is affected by treatment
# The parameters we estimate are 
#               Y_0, pi_r,   g_r,   g_s,  k_1,  sigma
lb = np.array([  0,    0,  0.00,  0.00, 0.00]) #, 10e-6])
ub = np.array([100,    1,  2e-1,  1e-1, 1e-0]) #, 10e4])
#lb = np.array([  0,    0,  0.00, -1e-0])
#ub = np.array([100,    1,  2e-1,  0.00])
# Y_0=50, pi_r=0.10, g_r=2e-3, g_s=1e-2, k_1=3e-2

# A training instance is a pair of history covariates X and estimated parameters Y
# Define minimum number of measurements for including period as training instance to X and Y
print("\nFinding right regions and estimating parameters...")
# Iterate through patients
# Identify valid periods and estimate parameters there: training_instance_dict = (training_instance_id, start, end, treatment_id). Y: (training_instance_id, parameters). 
# Find patients with periods that satisfy criteria of enough M protein measurements under chosen treatment 
# 1) Feature extract history up until period, put in X
# 2) Estimate parameters in period, put in Y
training_instance_id = 0
training_instance_dict = {} # A dictionary mapping training_instance_id to the patient name and the start and end of the interval with the treatment of interest 
Y_parameters = []
Y_increase_or_not = np.array([])
N_iter = 1000
minimum_number_of_measurements = 3
threshold_for_closeness_for_M_protein_at_start = 60

#treatment_id_of_interest = 15 # Dex+Len+Bor #COMMPASS_patient_dictionary["MMRF_1293"].treatment_history[5].id
# Iterate over all patients, look at their treatment periods one by one and check if it qualifies as a training item 
start_time = time.time()
how_many_regions = np.zeros(unique_treat_counter)
for name, patient in COMMPASS_patient_dictionary.items():
    if len(patient.measurement_times) > minimum_number_of_measurements:
        for outer_index, outer_treatment in enumerate(patient.treatment_history): # Outer loop so we pass each of them only once 
            if outer_treatment.id in [1,2,3,7,10,13,15,16]: # Subset of treatment ids we choose to include. [15,16,3,10,7,1,13,2]: #range(1,unique_treat_counter):
                treatment_id_of_interest = outer_treatment.id

                # Find periods of interest by looking through patient history 
                period_start = np.nan
                period_end = np.nan
                valid_interval = False
                dummy_measurement_times = np.array([])
                dummy_Mprotein_values = np.array([])
                dummy_Kappa_values = np.array([])
                dummy_Lambda_values = np.array([])
                this_history = np.array([])
                for index, treatment in enumerate(patient.treatment_history[outer_index:]): # Outer loop so we pass each of them only once 
                    if valid_interval == False: 
                        if treatment.id == treatment_id_of_interest:
                            # We found the start 
                            valid_interval = True
                            period_start = treatment.start
                            this_history = np.array([treatment])
                            # Find time M protein value closest in time to the start of the treatment
                            distances_to_treatment_start = abs(patient.measurement_times - np.repeat(treatment.start, len(patient.measurement_times)))
                            closest_index = np.argmin(distances_to_treatment_start)
                            if (not patient.measurement_times[closest_index] == treatment.start) and (min(distances_to_treatment_start) <= threshold_for_closeness_for_M_protein_at_start):
                                # Add that value as M protein value at treatment start if there is nothing there
                                if distances_to_treatment_start[closest_index] > 0:
                                    dummy_measurement_times = np.concatenate((patient.measurement_times[0:closest_index], [treatment.start], patient.measurement_times[closest_index:]))
                                    dummy_Mprotein_values = np.concatenate((patient.Mprotein_values[0:closest_index], [patient.Mprotein_values[closest_index]], patient.Mprotein_values[closest_index:]))
                                    dummy_Kappa_values = np.concatenate((patient.Kappa_values[0:closest_index], [patient.Kappa_values[closest_index]], patient.Kappa_values[closest_index:]))
                                    dummy_Lambda_values = np.concatenate((patient.Lambda_values[0:closest_index], [patient.Lambda_values[closest_index]], patient.Lambda_values[closest_index:]))
                                else:
                                    dummy_measurement_times = np.concatenate((patient.measurement_times[0:closest_index+1], [treatment.start], patient.measurement_times[closest_index+1:]))
                                    dummy_Mprotein_values = np.concatenate((patient.Mprotein_values[0:closest_index+1], [patient.Mprotein_values[closest_index]], patient.Mprotein_values[closest_index+1:]))
                                    dummy_Kappa_values = np.concatenate((patient.Kappa_values[0:closest_index+1], [patient.Kappa_values[closest_index]], patient.Kappa_values[closest_index+1:]))
                                    dummy_Lambda_values = np.concatenate((patient.Lambda_values[0:closest_index+1], [patient.Lambda_values[closest_index]], patient.Lambda_values[closest_index+1:]))
                            # Continute to next
                        #elif treatment.id == 0:
                        #    # The last M protein under this treatment might be the start
                        #    # Find time of last M protein value within period
                        #    valid_Mprotein = patient.Mprotein_values[patient.measurement_times>=treatment.start]
                        #    valid_times = patient.measurement_times[patient.measurement_times>=treatment.start]
                        #    valid_Mprotein = valid_Mprotein[valid_times<=treatment.end]
                        #    valid_times = valid_times[valid_times<=treatment.end]
                        #    if len(valid_times) > 0:
                        #        valid_interval = True
                        #        period_start = valid_times[-1]
                        #        zero_treatment = Treatment(period_start, treatment.end, treatment.id)
                        #        this_history = np.array([zero_treatment])
                        #        # Continute to next
                        else:
                            # Continue looking for the start
                            pass 
                    else: # if valid interval == True, then we are looking for 0 or the end 
                        if treatment.id == treatment_id_of_interest:
                            this_history = np.append(this_history,[treatment])
                            pass
                        elif treatment.id == 0:
                            # Only after the id of interest. We are extending the period by a drug holiday, then ending the period at the end of this treatment
                            if patient.name == "MMRF_1793" and treatment_id_of_interest == 15:
                                treatment = Treatment(treatment.start, 1770, treatment.id)
                            this_history = np.append(this_history,[treatment])
                            valid_interval = False
                            period_end = treatment.end
                            # Estimate parameters, add to Y and to the patient using the common function
                            training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, Y_increase_or_not = estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, Y_increase_or_not)
                            dummy_measurement_times = np.array([])
                            dummy_Mprotein_values = np.array([])
                            dummy_Kappa_values = np.array([])
                            dummy_Lambda_values = np.array([])
                        else: 
                            # We found the end at the beginning of this foreign treatment id
                            valid_interval = False
                            # Check if we captured the treatment of interest and not only zero: 
                            if treatment_id_of_interest in [element.id for element in this_history]:
                                period_end = treatment.start
                                # Estimate parameters, add to Y and to the patient using the common function
                                training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, Y_increase_or_not = estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, Y_increase_or_not)
                                dummy_measurement_times = np.array([])
                                dummy_Mprotein_values = np.array([])
                                dummy_Kappa_values = np.array([])
                                dummy_Lambda_values = np.array([])
                # After the last treatment, if ending on a valid interval, this is the end
                if valid_interval == True: 
                    # Check if we captured the treatment of interest and not only zero: 
                    if treatment_id_of_interest in [element.id for element in this_history]:
                        period_end = patient.treatment_history[-1].end
                        # Estimate parameters, add to Y and to the patient using the common function
                        training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, Y_increase_or_not = estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, Y_increase_or_not)

end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)
print("len(Y_increase_or_not):", len(Y_increase_or_not))
print("Y_increase_or_not == 1:", sum(Y_increase_or_not[Y_increase_or_not == 1]))
print("Y_increase_or_not == 0:", len(Y_increase_or_not) - np.count_nonzero(Y_increase_or_not))
print("Y_increase_or_not == np.nan:", sum(np.isnan(Y_increase_or_not)))
print("Y_increase_or_not something else:", sum([(elem not in [0,1]) for elem in Y_increase_or_not]) - sum(np.isnan(Y_increase_or_not)))
#COMMPASS_patient_dictionary["MMRF_1293"].print()

#print("Treatment id of interest:", treatment_id_of_interest)
#print("Number of regions with", minimum_number_of_measurements, "or more M protein measurements:", how_many_regions[treatment_id_of_interest])

np.save("training_instance_dict.npy", training_instance_dict)
###np.save("Y_parameters.npy", np.array(Y_parameters))

####print("Sort(enumerate(how_many_regions))", Sort(enumerate(how_many_regions)))
###counts_regions = np.flip(Sort(enumerate(how_many_regions)))
###for element in counts_regions[counts_regions[:,0]>1]:
###    patient_count = element[0]
###    treatment_id = element[1]
###    drug_names = get_drug_names_from_treatment_id_COMMPASS(treatment_id, treatment_id_to_drugs_dictionary_COMMPASS)
###    print("N =", patient_count, "   id = ", treatment_id, "    drugs:", drug_names)

# Save (training_instance_id, PUBLIC_ID, [start, end]) to allow df selection out of df_mprotein and df_drugs for feature extraction indexed by training_instance_id
###df_mprotein_and_dates = df_mprotein_and_dates[df_mprotein_and_dates['training_instance_id'].notna()]
df_mprotein_and_dates.reset_index(drop=True, inplace=True)
df_mprotein_and_dates.to_pickle("df_mprotein_and_dates.pkl")

###df_drugs_and_dates = df_drugs_and_dates[df_drugs_and_dates['training_instance_id'].notna()]
df_drugs_and_dates.reset_index(drop=True, inplace=True)
df_drugs_and_dates.to_pickle("df_drugs_and_dates.pkl")

# Plot estimate with observed values
#estimated_parameters_patient_2 = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=global_sigma)
#plot_treatment_region_with_estimate(parameters_patient_2, patient_2, estimated_parameters=estimated_parameters_patient_2, PLOT_ESTIMATES=True, plot_title="Patient 2")

#create a pickle file
picklefile = open('Y_parameters', 'wb')
#pickle the dictionary and write it to file
pickle.dump(np.array(Y_parameters), picklefile)
#close the file
picklefile.close()

picklefile = open('Y_increase_or_not', 'wb')
pickle.dump(Y_increase_or_not, picklefile)
picklefile.close()

picklefile = open('COMMPASS_patient_dictionary', 'wb')
pickle.dump(COMMPASS_patient_dictionary, picklefile)
picklefile.close()

