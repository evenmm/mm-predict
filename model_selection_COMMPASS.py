# Purposes of this script: 
#   Load COMMPASS_patient_dictionary
#   Add parameter estimates to the patients and create COMMPASS_patient_dictionary_with_estimates
#   Find sections with the right drug combination and enough data to perform inference
#   Perform inference of parameter set in each region
from utilities import *
warnings.simplefilter("ignore")

# Settings
N_iter = 1000 # separate minimzations of the least squares when fitting parameters
minimum_number_of_measurements = 3 # required number of M protein measurements for a period to be included in the dataset
threshold_for_closeness_for_M_protein_at_start = 60 # If M protein at period start is missing, it is imputed using the nearest measurement, but only if closer than this threshold number of days.
INCLUDE_SUBSEQUENT_DRUG_HOLIDAY = False #True # If a treatment is followed by a drug holiday, this decided if the holiday is included as part of the period

## Inference
# The length of ub and lb implicitly decides whether the effect of treatment is given a parameter or not. 
# If len(ub) = 4, then it is assumed that only periods under treatment are considered
# If len(ub) = 5, then k_1 models the effect of the drug on the sensitive population
# Simple exponential growth model with 2 populations, where only one is affected by treatment
# The parameters we estimate are 
#               Y_0, pi_r,   g_r,   g_s,  k_1,  sigma
#lb = np.array([  0,    0,  0.01,   0.00, 0.20]) #, 10e-6])
#ub = np.array([100,    1,  0.20,  lb[4], 1.00]) #, 10e4])
#lb = np.array([  0,    0,  0.00, -1e-0])
#ub = np.array([100,    1,  2e-1,  0.00])
# Y_0=50, pi_r=0.10, g_r=2e-3, g_s=1e-2, k_1=3e-2

# Bounds for model 1: 1 population, only resistant cells
#                Y_0,  g_r  sigma
lb_1 = np.array([  0, 0.001, 10e-6])
ub_1 = np.array([100, 0.20, 10e4])

# Bounds for model 2: 1 population, only sensitive cells
#                Y_0,   g_s,    k_1,  sigma
lb_2 = np.array([  0,   0.001,  0.20, 10e-6])
ub_2 = np.array([100, lb_2[2], 1.00, 10e4])

# Bounds for model 3: sensitive and resistant population
#                Y_0, pi_r,   g_r,   g_s,     k_1,  sigma
lb_3 = np.array([  0,    0.001,  0.001,   0.001,   0.20, 10e-6])
ub_3 = np.array([100,    1-0.001,  0.20,  lb_3[4], 1.00, 10e4])

# Load COMMPASS_patient_dictionary
picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary', 'rb')
COMMPASS_patient_dictionary = pickle.load(picklefile)
picklefile.close()

picklefile = open('./binaries_and_pickles/unique_treat_counter', 'rb')
unique_treat_counter = pickle.load(picklefile)
picklefile.close()

negative_loglikelihoods_1 = []
negative_loglikelihoods_2 = []
negative_loglikelihoods_3 = []
bic_values_1 = []
bic_values_2 = []
bic_values_3 = []
aic_c_values_1 = []
aic_c_values_2 = []
aic_c_values_3 = []
chosen_models = []

# Find the treatment id of the required treatment, extract those treatment regions and perform inference there
def estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3):
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
        print("Saving a case from", patient.name, "- treatment id", treatment_id_of_interest, "- training instance id", training_instance_id)
        how_many_regions[treatment_id_of_interest] = how_many_regions[treatment_id_of_interest] + 1
        # Note the time limits of this period: Last M protein measurement while still on treatment
        last_measurement_time_on_treatment = valid_times[-1]
        # Estimate parameters for a dummy patient within this interval
        dummmy_patient = COMMPASS_Patient(measurement_times=valid_times, drug_dates=[], drug_history=[], treatment_history=this_history, Mprotein_values=valid_Mprotein, Kappa_values=valid_Kappa, Lambda_values=valid_Lambda, covariates=[], name="dummy")
        this_patient = dummmy_patient

        # Estimate parameters for model 1
        # Model 1: exp rho t            (2+1=3 parameters: Y0, rho, sigma)
        this_estimate_m1 = estimate_drug_response_parameters_any_model(this_patient, lb_1, ub_1, N_iterations=N_iter) #, sigma_noise_std=1
        array_x = np.array([this_estimate_m1.Y_0, this_estimate_m1.g_r, this_estimate_m1.sigma])
        predictions = measure_Mprotein_noiseless(this_estimate_m1, this_patient.measurement_times, this_patient.treatment_history)
        sumofsquares_model_1 = np.sum((this_patient.Mprotein_values - predictions)**2)
        sample_variance_unadjusted_model_1 = sumofsquares_model_1/len(this_patient.measurement_times)
        negative_loglikelihoods_1.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_1))
        #print(negative_loglikelihoods_1[training_instance_id])
        
        # Estimate parameters for model 2
        # Model 2: exp t(alpha - k)     (3+1=4 parameters: Y0, alpha, K, sigma)
        this_estimate_m2 = estimate_drug_response_parameters_any_model(this_patient, lb_2, ub_2, N_iterations=N_iter) #, sigma_noise_std=1
        array_x = np.array([this_estimate_m2.Y_0, this_estimate_m2.g_s, this_estimate_m2.k_1, this_estimate_m2.sigma])
        predictions = measure_Mprotein_noiseless(this_estimate_m2, this_patient.measurement_times, this_patient.treatment_history)
        sumofsquares_model_2 = np.sum((this_patient.Mprotein_values - predictions)**2)
        sample_variance_unadjusted_model_2 = sumofsquares_model_2/len(this_patient.measurement_times)
        negative_loglikelihoods_2.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_2))
        #print(negative_loglikelihoods_2[training_instance_id])
        
        # Estimate parameters for model 3
        # Model 3: Both together.       (5+1=6 parameters: Y0, pi, rho, alpha, K, sigma)
        this_estimate_m3 = estimate_drug_response_parameters_any_model(this_patient, lb_3, ub_3, N_iterations=N_iter) #, sigma_noise_std=1
        array_x = np.array([this_estimate_m3.Y_0, this_estimate_m3.pi_r, this_estimate_m3.g_r, this_estimate_m3.g_s, this_estimate_m3.k_1, this_estimate_m3.sigma])
        predictions = measure_Mprotein_noiseless(this_estimate_m3, this_patient.measurement_times, this_patient.treatment_history)
        sumofsquares_model_3 = np.sum((this_patient.Mprotein_values - predictions)**2)
        sample_variance_unadjusted_model_3 = sumofsquares_model_3/len(this_patient.measurement_times)
        negative_loglikelihoods_3.append(negative_loglikelihood_any_model(array_x, this_patient)) #, sigma_noise_std=sample_variance_unadjusted_model_3))
        #print(negative_loglikelihoods_3[training_instance_id])
        
        N_observations_this_period = len(dummmy_patient.measurement_times)
        # BIC values: k*ln(N) + 2*negative loglikelihood
        bic_values_1.append(len(lb_1)*np.log(N_observations_this_period) + 2*negative_loglikelihoods_1[training_instance_id])
        bic_values_2.append(len(lb_2)*np.log(N_observations_this_period) + 2*negative_loglikelihoods_2[training_instance_id])
        bic_values_3.append(len(lb_3)*np.log(N_observations_this_period) + 2*negative_loglikelihoods_3[training_instance_id])
        #aic_c_values_1.append(2*len(lb_1) + 2*negative_loglikelihoods_1[training_instance_id] + 2*len(lb_1)*(len(lb_1)+1)/(N_observations_this_period - len(lb_1) - 1))
        #aic_c_values_2.append(2*len(lb_2) + 2*negative_loglikelihoods_2[training_instance_id] + 2*len(lb_2)*(len(lb_2)+1)/(N_observations_this_period - len(lb_2) - 1))
        #aic_c_values_3.append(2*len(lb_3) + 2*negative_loglikelihoods_3[training_instance_id] + 2*len(lb_3)*(len(lb_3)+1)/(N_observations_this_period - len(lb_3) - 1))
        bic_values_this_patient = bic_values_1[training_instance_id], bic_values_2[training_instance_id], bic_values_3[training_instance_id]
        #aic_c_values_this_patient = aic_c_values_1[training_instance_id], aic_c_values_2[training_instance_id], aic_c_values_3[training_instance_id]
        model_choice = bic_values_this_patient.index(min(bic_values_this_patient)) + 1
        chosen_models.append(model_choice)
        parameter_estimates_this_patients = [this_estimate_m1, this_estimate_m2, this_estimate_m3]
        this_estimate = parameter_estimates_this_patients[model_choice-1]
        
        #this_estimate = estimate_drug_response_parameters(dummmy_patient, lb, ub, N_iterations=N_iter)
        training_instance_dict[training_instance_id] = [patient.name, period_start, period_end, treatment_id_of_interest, last_measurement_time_on_treatment, this_estimate]
        # Add estimates to Y_parameters
        Y_parameters.append(this_estimate) # training_instance_id is position in Y_parameters
        #patient.add_parameter_estimate(this_estimate, (period_start, period_end), dummmy_patient) # NOt really used, it is in training_instance_dict
        training_instance_id = training_instance_id + 1

        # Plotting treatment region with estimate (using model 3 estimate)
        #for index, param_set in enumerate(patient.parameter_estimates):
        estimated_parameters = this_estimate # patient.parameter_estimates[index]
        plot_patient = dummmy_patient # patient.dummmy_patients[index]
        #savename = "./COMMPASS_estimate_plots/treatment_id_"+str(treatment_id_of_interest)+"/Treatment_"+str(treatment_id_of_interest)+"_"+patient.name+"_at_time"+str(period_start)+"Y_0="+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sigma="+str(estimated_parameters.sigma)+".png"
        savename = "./COMMPASS_estimate_plots/model_selection/model_"+str(model_choice)+"/Model_"+str(model_choice)+"_id_"+str(training_instance_id)+"_Treatment_"+str(treatment_id_of_interest)+"_"+patient.name+"_at_time"+str(period_start)+"Y_0="+str(estimated_parameters.Y_0)+", pi_r="+str(estimated_parameters.pi_r)+", g_r="+str(estimated_parameters.g_r)+", g_s="+str(estimated_parameters.g_s)+", k_1="+str(estimated_parameters.k_1)+", sigma="+str(estimated_parameters.sigma)+".png"
        plot_title = patient.name
        plot_treatment_region_with_estimate(estimated_parameters, plot_patient, estimated_parameters=[], PLOT_ESTIMATES=False, plot_title=plot_title, savename=savename)
    return training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3

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

#treatment_id_of_interest = 15 # Dex+Len+Bor #COMMPASS_patient_dictionary["MMRF_1293"].treatment_history[5].id
# Iterate over all patients, look at their treatment periods one by one and check if it qualifies as a training item 
start_time = time.time()
how_many_regions = np.zeros(unique_treat_counter)
for name, patient in COMMPASS_patient_dictionary.items():
    if len(patient.measurement_times) > minimum_number_of_measurements:
        treatment_id_list = [1,2,3,7,10,13,15,16] #[38,62,66,110] are other combinations with the same drugs (but 66 only has 6 patients). # 30 for patient 1727 # Subset of treatment ids we choose to include. [15,16,3,10,7,1,13,2]: #range(1,unique_treat_counter):
        for outer_index, outer_treatment in enumerate(patient.treatment_history): # Outer loop so we pass each of them only once 
            if outer_treatment.id in treatment_id_list:
                treatment_id_of_interest = outer_treatment.id

                # New, simpler and faster version when not including treatment holidays
                # Check if period satisfies, if so then add it as a training case
                # *Just check the current period instead of looping, it is faster*
                # *Then remove treatment_id_list popping at end
                # Currently the pop remains as there were duplicates: Now there is at most one case per patient

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
                            training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3 = estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3)
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
                                training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3 = estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3)
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
                        training_instance_id, how_many_regions, training_instance_dict, patient, Y_parameters, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3 = estimate_and_save_region_estimate(training_instance_id, period_start, period_end, minimum_number_of_measurements, dummy_measurement_times, dummy_Mprotein_values, dummy_Kappa_values, dummy_Lambda_values, how_many_regions, training_instance_dict, patient, Y_parameters, treatment_id_of_interest, negative_loglikelihoods_1, negative_loglikelihoods_2, negative_loglikelihoods_3, bic_values_1, bic_values_2, bic_values_3, aic_c_values_1, aic_c_values_2, aic_c_values_3)
                treatment_id_list.remove(treatment_id_of_interest)

end_time = time.time()
time_duration = end_time - start_time
print("Time elapsed:", time_duration)
#COMMPASS_patient_dictionary["MMRF_1293"].print()

#print("Treatment id of interest:", treatment_id_of_interest)
#print("Number of regions with", minimum_number_of_measurements, "or more M protein measurements:", how_many_regions[treatment_id_of_interest])

#np.save("./binaries_and_pickles/training_instance_dict.npy", training_instance_dict)
#print(len(training_instance_dict.values()))
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

##create a pickle file
#picklefile = open('./binaries_and_pickles/Y_parameters', 'wb')
##pickle the dictionary and write it to file
#pickle.dump(np.array(Y_parameters), picklefile)
##close the file
#picklefile.close()

#picklefile = open('./binaries_and_pickles/COMMPASS_patient_dictionary_with_estimates', 'wb')
#pickle.dump(COMMPASS_patient_dictionary, picklefile)
#picklefile.close()
