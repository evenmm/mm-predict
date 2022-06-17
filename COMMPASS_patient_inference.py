# Purposes of this script: 
#   Load COMMPASS_patient_dictionary
#   Find sections with the right drug combination and enough data to perform inference
#   Perform inference of parameter set in each region
from utilities import *

# Settings
N_iter = 1000 # separate minimzations of the least squares when fitting parameters
minimum_number_of_measurements = 3 # required number of M protein measurements for a period to be included in the dataset
threshold_for_closeness_for_M_protein_at_start = 60 # If M protein at period start is missing, it is imputed using the nearest measurement, but only if closer than this threshold number of days.
INCLUDE_SUBSEQUENT_DRUG_HOLIDAY = True # If a treatment is followed by a drug holiday, this decided if the holiday is included as part of the period

## Inference
# The length of ub and lb implicitly decides whether the effect of treatment is given a parameter or not. 
# If len(ub) = 4, then it is assumed that only periods under treatment are considered
# If len(ub) = 5, then k_1 models the effect of the drug on the sensitive population
# Simple exponential growth model with 2 populations, where only one is affected by treatment
# The parameters we estimate are 
#               Y_0, pi_r,   g_r,   g_s,  k_1,  sigma
lb = np.array([  0,    0,  0.00,   0.00, 0.20]) #, 10e-6])
ub = np.array([100,    1,  0.20,  lb[4], 1.00]) #, 10e4])
#lb = np.array([  0,    0,  0.00, -1e-0])
#ub = np.array([100,    1,  2e-1,  0.00])
# Y_0=50, pi_r=0.10, g_r=2e-3, g_s=1e-2, k_1=3e-2


# Load COMMPASS_patient_dictionary
picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/unique_treat_counter', 'rb')
unique_treat_counter = pickle.load(picklefile)
picklefile.close()

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
    if len(valid_times) >= minimum_number_of_measurements and max(valid_Mprotein) > 0:
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

#treatment_id_of_interest = 15 # Dex+Len+Bor #COMMPASS_patient_dictionary["MMRF_1293"].treatment_history[5].id
# Iterate over all patients, look at their treatment periods one by one and check if it qualifies as a training item 
start_time = time.time()
how_many_regions = np.zeros(unique_treat_counter)
for name, patient in COMMPASS_patient_dictionary.items():
    if len(patient.measurement_times) > minimum_number_of_measurements:
        for outer_index, outer_treatment in enumerate(patient.treatment_history): # Outer loop so we pass each of them only once 
            if outer_treatment.id in [1,2,3,7,10,13,15,16]: #[38,62,66,110] are other combinations with the same drugs (but 66 only has 6 patients). # 30 for patient 1727 # Subset of treatment ids we choose to include. [15,16,3,10,7,1,13,2]: #range(1,unique_treat_counter):
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
                for index, treatment in enumerate(patient.treatment_history[outer_index:]): # Inner loop to check what happens after this exact treatment from the outer loop
                    if valid_interval == False: 
                        if treatment.id == treatment_id_of_interest:
                            # We found the start 
                            valid_interval = True
                            period_start = treatment.start
                            this_history = np.array([treatment])
                            # Find the M protein value closest in time to the start of the treatment
                            distances_to_treatment_start = patient.measurement_times - np.repeat(treatment.start, len(patient.measurement_times))
                            abs_distances_to_treatment_start = abs(distances_to_treatment_start)
                            closest_index = np.argmin(abs_distances_to_treatment_start)
                            if (not abs_distances_to_treatment_start[closest_index] == 0) and (min(abs_distances_to_treatment_start) <= threshold_for_closeness_for_M_protein_at_start):
                                # Add that value as M protein value at treatment start if there is nothing there
                                if distances_to_treatment_start[closest_index] > 0: # Treatment.start comes before the value that was closest
                                    dummy_measurement_times = np.concatenate((patient.measurement_times[0:closest_index], [treatment.start], patient.measurement_times[closest_index:]))
                                    dummy_Mprotein_values = np.concatenate((patient.Mprotein_values[0:closest_index], [patient.Mprotein_values[closest_index]], patient.Mprotein_values[closest_index:]))
                                    dummy_Kappa_values = np.concatenate((patient.Kappa_values[0:closest_index], [patient.Kappa_values[closest_index]], patient.Kappa_values[closest_index:]))
                                    dummy_Lambda_values = np.concatenate((patient.Lambda_values[0:closest_index], [patient.Lambda_values[closest_index]], patient.Lambda_values[closest_index:]))
                                else: # Treatment.start comes after the value that was closest. Indexing is ok, even in empty space
                                    dummy_measurement_times = np.concatenate((patient.measurement_times[0:closest_index+1], [treatment.start], patient.measurement_times[closest_index+1:]))
                                    dummy_Mprotein_values = np.concatenate((patient.Mprotein_values[0:closest_index+1], [patient.Mprotein_values[closest_index]], patient.Mprotein_values[closest_index+1:]))
                                    dummy_Kappa_values = np.concatenate((patient.Kappa_values[0:closest_index+1], [patient.Kappa_values[closest_index]], patient.Kappa_values[closest_index+1:]))
                                    dummy_Lambda_values = np.concatenate((patient.Lambda_values[0:closest_index+1], [patient.Lambda_values[closest_index]], patient.Lambda_values[closest_index+1:]))
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
                        elif treatment.id == 0 and INCLUDE_SUBSEQUENT_DRUG_HOLIDAY == True:
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
print("Number of intervals:", len(Y_increase_or_not))
print("Number of 0s:", len(Y_increase_or_not) - np.count_nonzero(Y_increase_or_not))
print("Number of 1s:", sum(Y_increase_or_not[Y_increase_or_not == 1]))
print("Number of nans:", sum(np.isnan(Y_increase_or_not)))
print("Number of other things:", sum([(elem not in [0,1]) for elem in Y_increase_or_not]) - sum(np.isnan(Y_increase_or_not)))
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

# Plot estimate with observed values
#estimated_parameters_patient_2 = Parameters(Y_0=best_x[0], pi_r=best_x[1], g_r=best_x[2], g_s=best_x[3], k_1=best_x[4], sigma=global_sigma)
#plot_treatment_region_with_estimate(parameters_patient_2, patient_2, estimated_parameters=estimated_parameters_patient_2, PLOT_ESTIMATES=True, plot_title="Patient 2")

#create a pickle file
picklefile = open('./binaries_and_pickles/Y_parameters', 'wb')
#pickle the dictionary and write it to file
pickle.dump(np.array(Y_parameters), picklefile)
#close the file
picklefile.close()

picklefile = open('./binaries_and_pickles/Y_increase_or_not', 'wb')
pickle.dump(Y_increase_or_not, picklefile)
picklefile.close()
